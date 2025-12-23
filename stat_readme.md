下面按“阶段 1 / 阶段 1.5”的目标—动作—问题—解决方案—最终产出，做一次尽量详细的复盘（以你这条 AGAR 项目线为准）。

---

## 阶段 1：把真实 AGAR 数据接进训练流水线（以 FiftyOne 为数据源）

### 1) 阶段目标

* 把你已经在 FiftyOne 里构建好的原始数据集（`AGARlower`/`agar_lower`）接入训练，使训练代码能够从真实图片与 bbox 标注产出 PyTorch Detection 任务所需的 `(image, target)`。
* 目标最低要求是：**能稳定 iterate DataLoader**（哪怕暂时不训练），并且 target 满足 torchvision detection 的 contract：

  * `target["boxes"]`: `FloatTensor[N,4]`，xyxy
  * `target["labels"]`: `LongTensor[N]`
  * 允许 `N=0`（无框样本）

### 2) 你们做了哪些具体工作

1. **实现/明确了 FiftyOneDataset 的读取逻辑**

   * 从 `sample.filepath` 读图
   * 从 `sample.detections.detections` 里取 `det.bounding_box`（FiftyOne 的归一化 xywh）
   * 转为像素级 xyxy
   * label 用 `class_to_idx` 映射到 int 类别
   * 无框时输出 `zeros((0,4))` 和 `zeros((0,))`

2. **尝试在导出脚本 fo_to_coco 中使用 train/val tag 视图**

   * 使用：

     * `--train-view "tag:train"`
     * `--val-view "tag:val"`

### 3) 遇到的关键问题

**问题 A：train/val tag 不存在**

* 你检查后发现：

  * `match_tags('train'): 0`
  * `match_tags('val'): 0`
* 说明你的数据集虽然有很多 tag（类别 tag 等），但并没有写入 `train` / `val` 这样的拆分标签。

**问题 B：导出结果为 0 images / 0 annotations**

* 由于 train/val view 为空，导出自然全是 0：

  * `train_images: 0`
  * `val_images: 0`
  * annotations 都是 0

**问题 C：MongoDB “oldest supported major version” 警告**

* FiftyOne 提醒数据库版本太老。它当时只是警告，但会干扰判断（让人误以为导出失败是 DB 的问题）。

### 4) 怎么解决的

1. **不依赖 tag 进行 split**

   * 修改导出器逻辑，支持 `--split-method random`
   * 即：对样本集合 shuffle，再 take/skip 切分为 train/val
   * 好处：不需要写 tag，不需要改 FiftyOne 数据库里的样本元信息

2. **修正对 SampleView 的 label field 访问**

   * 导出器要能在 `SampleView` 上安全读取 `detections` 字段（避免 `getattr`/`get_field` 差异导致读不到）

3. **明确 MongoDB 警告不阻塞导出**

   * 保持警告即可，或者可在 FiftyOne config 里关 `database_validation` 来静默（你们也记录了这点，但没有强制要求立刻改 DB）

### 5) 阶段 1 的最终产出

* 成功从 FO 数据集导出 COCO（即便当时还在阶段 1 的 FO 语境里，实际上已为 1.5 铺路）：

  * `Train images: 3544, annotations: 68581`
  * `Val images: 887, annotations: 17895`
* 证明：数据标注读取与 bbox 转换是对的、数据量吻合预期。

---

## 阶段 1.5：去耦合 FiftyOne——导出 COCO cache 并以 COCO 作为训练数据源

> 你明确提出“不想每次都与 fiftyone 耦合在一起”，因此 1.5 的核心是：**一次性导出 + 后续纯文件训练**。

### 1) 阶段目标

* 从 FiveOne 数据集中导出一个可复用的 COCO 缓存目录（images_dir + train.json/val.json）
* 训练/评估/推理都不再依赖 FO 和 MongoDB
* 数据管道完全文件化，便于：

  * 复现实验
  * 迁移到别的机器/环境
  * debug（COCO 是行业标准）

### 2) 你们做了哪些具体工作（工程化落地）

1. **确定 COCO cache 目录与参数**

   * 例：

     * `images_dir=/home/gh/dataset/loweragar/data`
     * `out_dir=/home/gh/dataset/loweragar/coco_cache/agar_lower_coco`
     * `train_json=/home/gh/dataset/loweragar/coco_cache/agar_lower_coco/coco/train.json`
     * `val_json=/home/gh/dataset/loweragar/coco_cache/agar_lower_coco/coco/val.json`

2. **在 Hydra 配置中引入 data.source 分支**

   * `data.source: coco` 作为默认
   * datamodule 能根据 source 选择构建 COCO dataset / （未来也可扩展回 fiftyone）

3. **新增 COCO inspector 工具**

   * `python -m agar.data.inspect_coco --json ... --images-dir ...`
   * 输出：

     * images / annotations / categories 数量
     * category_id 分布
     * bbox 抽样检查
     * `--check-all` 做全量 bbox 合法性统计（不读取图片也可）

4. **bbox 处理进一步加固**

   * 训练 dataset 中对 bbox 做 clip 到图像边界，避免 out-of-bounds 触发 torchvision/metric 崩溃
   * inspector 报告 out_of_bounds_ratio（以及 check_all 的统计）

5. **新增 smoke runner**

   * `python -m agar.smoke experiment=...`
   * 目的：快速跑通 1 个 batch/1 次 optimizer step，验证数据->模型->loss 这条链路不崩

### 3) 阶段 1.5 遇到的关键问题与解决

**问题 A：Hydra experiment 覆盖失败**

* 报错：

  * `Could not override 'experiment'. No match in the defaults list`
  * 后续又出现：`Could not override 'data@experiment.data'`

**解决：**

1. 在 `config.yaml` 的 defaults 里把 experiment 组纳入搜索路径
2. 把每个 experiment yaml 改为 `@package _global_`，使其中的 `data/model/train/eval` 覆盖发生在 root，而不是挂在 `experiment.*` 子树上
3. 最终 `python -m agar.train experiment=... --cfg job` 能正确打印 merged config（你贴出来的就是合并后的 root config）

---

**问题 B：Fabric 意外启动 2-GPU DDP，rank1 SIGSEGV**

* 你本意单卡，但 Fabric/Lightning path 自动按 `torch.cuda.device_count()` 推成 2 卡
* 导致 rank1 崩溃（-11），训练直接被 kill

**解决：**

* 在配置里显式加：

  * `train.devices: 1`
  * `train.accelerator/strategy: auto`（但 devices=1 关键）
* train/eval/smoke 等入口都读取这些字段构造 Fabric，避免隐式 2-GPU spawn
* 并增加 `launch_check` 工具确认：

  * `cuda.device_count=2`
  * `fabric.world_size=1`
  * `cuda.current_device=0`
  * 即：机器有两卡，但训练 world_size 固定 1

---

**问题 C：MAP 指标依赖缺失导致评估崩溃**

* 报错：

  * `MeanAveragePrecision requires pycocotools or faster-coco-eval`

**解决：**

1. README 增加 “COCO eval dependencies” 安装说明
2. 评估逻辑做 graceful fallback：缺依赖就提示并跳过（或建议 `train.eval_every=0`）
3. 你实际跑到末尾后能 “Done”，只是 `best_metric=-inf`（因为跳过 eval）

---

**问题 D：batch_size 变大但速度不明显**

* 你观察到 batch_size=16 但速度提升不大，并且 first_batch data_time 有时会更高
* 这属于“数据加载与模型 compute 的瓶颈混合问题”，不是 bug

**应对措施（当时做的动作）：**

* 增加 `first_batch profiling`：

  * data_time / fwd_time / bwd_time / opt_time
* 增加 `max_steps` early exit + `exp_tiny_steps` 快速实验配置，避免每次 full epoch 才知道哪里慢

---

### 4) 阶段 1.5 最终产出（你现在已经具备的能力）

* **数据层**：FO -> COCO cache 成功，训练不依赖 FO/MongoDB
* **工具链**：

  * `inspect_coco`：可视化检查 COCO 是否健康
  * `smoke`：一键检查训练链路可跑
  * `launch_check`：验证单卡/多卡启动逻辑
* **训练层**：

  * 单卡 AMP 可稳定跑完并输出 loss
  * DDP/NCCL 通过“固定 NCCL socket 网卡 env + run_ddp.sh”已经能启动并跑 `max_steps=5`（你最后这条日志就是证据）
* **工程层**：

  * Hydra experiment 机制已修复，可以用 `experiment=...` 管理所有实验
  * bbox clip、防越界统计、早停与 profiling 都已具备

---

## 一句话概括

* **阶段 1**：从 FO 直接读数据的路线在“split/tag”上踩坑，最终通过“random split + 修正 label field 读取”把导出打通。
* **阶段 1.5**：把 FO 彻底去耦合成 COCO cache，并在工程上把 Hydra/Fabric/metric/DDP 这套现代化训练栈的坑基本踩平，最终达到“单卡稳定训练 + 多卡可启动并执行 step”的可用状态。

如果你愿意，我还可以把这份总结进一步改写成你 README 的“Milestone/Changelog”风格，或者写成给组会汇报的 1 页版本（包含关键命令与关键错误截图点）。
