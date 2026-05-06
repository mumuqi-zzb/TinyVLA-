# VLA的项目怎么训练？
### 第一步：准备“食材”（物料准备）

这是最耗时、最占硬盘的一步。你不能“凭空”训练，必须准备好以下三样东西：

1. **一个预训练好的 VLM（基座模型）**
    
    - **是什么**：比如 LLaVA、Qwen-VL、Prismatic-VL。它们已经具备了极强的视觉理解和逻辑推理能力。
        
    - **你的现状**：你刚才费劲下载的 `Llava-Pythia-700M` 就是这个基座。
        
2. **机器人专家数据集（最核心）**
    
    - **是什么**：这不是普通的图片或文本，而是“轨迹（Trajectories）”。数据通常是以 `.hdf5` 或 `.zarr` 格式存储的。
        
    - **内容包括**：
        
        - `Image`：机器人多个视角的摄像头画面（比如正前方、机械臂手腕处）。
            
        - `Language`：人类给的指令（例如："pick up the red apple"）。
            
        - `State`：机器人当前的本体状态（各个关节的角度、夹爪的开合度）。
            
        - `Action`：人类遥控机器人完成任务时，下一步真实的移动坐标（专家动作）。
            
    - **来源**：常用的有 Open X-Embodiment (OXE) 混合数据集、ALOHA 数据集、DROID（Franka 机器人）数据集等。这动辄几百 GB 甚至上 TB。
        
3. **算力（GPU）**
    
    - VLA 模型的显存占用极大。微调 700M 的模型可能单张 4090 勉强够用，但训练 7B（70 亿参数）级别的 OpenVLA，通常需要 8 张 80G 的 A100。
        

---

### 第二步：改造“大脑”（架构设计）

原来的大语言模型只能输出“文字（Token）”，怎么让它输出“动作（Action）”呢？这需要给它动个小手术。

- **加装“动作头”（Action Head）**： 在模型的最后一层，切断输出文字的通道，接上一个专门输出机械臂坐标的神经网络。
    
    - **连续动作流派**：比如 TinyVLA 用的 **Diffusion（扩散模型）头**，或者 Pi0 用的 **Flow Matching**，它们直接输出精确的连续坐标系数值。
        
    - **离散动作流派**：比如 OpenVLA 和 RT-2，它们把物理世界的坐标划分成一个个“格子”，把动作当成“生僻字”让大模型像写文章一样预测出来（Autoregressive）。
        

---

### 第三步：正式“炒菜”（训练过程）

当代码、数据、模型都就位后，你在终端按下 `bash train.sh`，程序会经历以下流程：

1. **数据预检与归一化（这就是你卡住的地方！）**
    
    - 程序会先把那几百 GB 的 `.hdf5` 数据集全部扫一遍。
        
    - **目的**：算出机械臂在所有任务中的**最大动作范围、最小动作范围、平均值和方差**。
        
    - **产出**：把这些统计好的数字，存成一个字典——也就是你苦苦寻找的 **`dataset_stats.pkl`**。这是模型理解现实世界物理尺度的“换算尺”。
        
2. **冻结与微调（LoRA）**
    
    - 为了节省算力和防止模型“变傻”（忘记怎么认东西），通常会冻结（Freeze）视觉编码器和语言主干的大部分参数。
        
    - 只训练新加的“动作头”，并在语言模型中插入 **LoRA** 旁路进行轻量化微调。
        
3. **前向传播与计算损失**
    
    - **输入**：把当前帧的图片 + "抓起苹果"的指令喂给模型。
        
    - **输出**：模型预测出一个动作（比如：向前移动 0.1米，夹爪闭合）。
        
    - **纠错（Loss）**：把模型预测的动作，和数据集中人类专家的真实动作进行对比，计算误差（MSE Loss），然后反向传播更新权重。
        
    - 这个过程要在几百万条数据上循环成千上万次（Steps）。
        

---

### 第四步：下地干活（部署与评估 Eval）

也就是你现在手头正在跑的 `eval_real_franka_6.py`。

- 训练结束后，我们把炼好的权重文件和那把“换算尺”（`.pkl` 文件）一起拷贝到真机或仿真环境（如 MuJoCo）中。
    
- 机器人看着实时的摄像头画面，听着你的文本指令，实时推断出下一步动作。

# TinyVLA的训练
这一章讲的是“整场训练是怎么被真正启动起来的”。

在这个仓库里，训练不是直接手敲一大串参数，而是通常从脚本 [scripts/train.sh:3-61](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/scripts/train.sh#L3-L61) 启动。这里先定义了 `ACTION_HEAD` 和 `OUTPUT`，然后通过 `deepspeed` 调用 [train_tinyvla.py](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py)。也就是说，shell 脚本负责的是“把训练命令拼出来”，而 Python 入口负责的是“真正执行训练逻辑”。

真正的 Python 主入口在 [train_tinyvla.py:287-300](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L287-L300)。这里先调用 `parse_pythia()` 解析参数，再把解析后的参数打包成 `config`，最后进入 `main(config=config, llava_pythia_config=llava_pythia_config)`。这意味着从系统视角看，训练流程的第一跳是：

1. `scripts/train.sh` 组装命令
2. `train_tinyvla.py` 读取命令行参数
3. `main()` 正式进入训练链路

这一章的意义，是先把“训练从哪儿开始”讲清楚，否则后面的数据、模型、损失都会显得像凭空出现。
### 第一阶段：外层指挥官 (`scripts/train.sh`)

训练的绝对起点是这个 Shell 脚本。当你运行这个脚本时，它主要做了三件“后勤”工作：

- **定义输出路径**：它首先确定了模型保存的文件夹路径 `OUTPUT=/path/to/save_dir`，并在发现该文件夹不存在时主动创建它。
    
- **备份训练配置**：它会执行 `cp ./scripts/train.sh $OUTPUT`，把当前的启动脚本备份到输出文件夹里，防止你以后忘了这次训练用了什么参数。
    
- **下达长串指令并启动引擎**：它使用了 `deepspeed` 工具来启动分布式的 Python 训练程序。最关键的是，它通过命令行向 Python 传递了大量的超参数，比如任务名称 `--task_name "example_task_config"`、学习率 `--learning_rate 2e-4` 以及动作头类型 `--action_head_type $ACTION_HEAD` 等。
### 第二阶段：内层执行者 (`train_tinyvla.py`)

当 Shell 脚本用 `deepspeed` 唤醒 Python 脚本后，真正的代码逻辑从 `if __name__ == '__main__':` 这一行开始执行。它的启动可以拆分为四个标准动作：

1. **解析指令 (`parse_pythia`)**：程序首先调用 `parse_pythia()` 函数，把 Shell 脚本里那一大串像 `--lora_enable True` 这样的文字参数，分门别类地转化为 Python 内部可以识别的对象（比如模型参数、数据参数、训练参数等）。
    
2. **打包配置 (`config`)**：将解析好的所有参数打包进一个名为 `config` 的大字典中，方便后续在各个函数之间传递。
    
3. **进入主流程 (`main`)**：带着这个装满参数的 `config` 字典，程序正式进入 `main(config=config, llava_pythia_config=llava_pythia_config)` 函数。
    
4. **准备数据与开始训练**：在 `main()` 函数内部，程序会先根据配置加载数据 (`load_data`)，然后再调用 `train_bc()` 函数正式开始模型训练。

## Chapter 2. 参数登场：命令行参数如何进入系统

这一章讲的是外部传入的训练参数，最后是怎么变成程序内部配置对象的。

在 [train_tinyvla.py:26-112](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L26-L112) 中，代码把参数分成四类：

- `ActionArguments`：动作头类型、动作维度、状态维度、chunk 大小
- `ModelArguments`：基础模型路径、版本、图像 token 设置
- `DataArguments`：任务名、图像长宽比、是否跳过镜像数据
- `TrainingArguments`：学习率、保存频率、训练步数、LoRA 开关、量化位数等

在 [train_tinyvla.py:121-172](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L121-L172) 的 `parse_pythia()` 中，这四类参数通过 `HfArgumentParser` 被解析成 dataclass 实例。随后程序还做了两件关键的事：

1. 根据 `fp16/bf16` 推导训练使用的计算精度
2. 把 `ActionArguments` 里的字段逐项写进 `LlavaPythiaConfig`

这意味着动作头的结构参数不是只保留在命令行层，而是被真正注入到了模型配置内部。

**关键代码位置**

- [train_tinyvla.py:27-31](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L27-L31)
- [train_tinyvla.py:34-40](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L34-L40)
- [train_tinyvla.py:43-48](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L43-L48)
- [train_tinyvla.py:51-112](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L51-L112)
- [train_tinyvla.py:121-172](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L121-L172)
### 参数怎么理解：
“命令行参数”（Command-line arguments）听起来像个高深的技术词汇，但其实非常生活化。

你可以把你的 Python 训练程序（`train_tinyvla.py`）想象成一台**全自动洗衣机**。

- 你不能只对洗衣机喊一句“开始洗衣服”，你得在面板上设定参数：水温多少度、洗多少分钟、要不要甩干。
    
- **命令行参数**，就是你在启动程序时，在终端（Shell）里敲进去的这些“设定选项”。
    
#### 结合 Chapter 2 拆解一下：

我们在启动训练时，通常会在终端里输入长长的一串命令，比如： `python train_tinyvla.py --learning_rate 2e-4 --action_head_type droid_diffusion --freeze_vision_tower True`

在 Chapter 2 中，这段代码要解决的核心问题就是：**外部传入的训练参数，最后是怎么变成程序内部配置对象的**。程序内部是看不懂那行长长的文本命令的，它需要把这些“命令行参数”分门别类地装进自己认识的盒子里。

具体流程如下：

1. **分门别类（建立四个盒子）：** 代码预先定义了四个数据类（盒子）来接收这些参数：
    
    - `ActionArguments`（动作参数）：管机器人的手，比如动作头类型、动作维度等。
        
    - `ModelArguments`（模型参数）：管模型的大脑，比如基础模型路径、版本等。
        
    - `DataArguments`（数据参数）：管怎么吃数据，比如任务名、图像长宽比等。
        
    - `TrainingArguments`（训练参数）：管训练过程，比如学习率、训练步数、LoRA 开关等。
        
2. **自动分发（解析参数）：** 程序调用了一个叫 `parse_pythia()` 的函数，利用 `HfArgumentParser` 这个工具，把你在命令行里敲的那些参数自动“抓”出来，并对号入座放进上面那四个盒子里（解析成 dataclass 实例）。
    
3. **深度注入（注入核心配置）：** 参数不仅停留在表面，程序还会把 `ActionArguments` 里的字段逐项写进模型的核心配置 `LlavaPythiaConfig` 中。这意味着，你从命令行传进来的动作头结构参数，被真正注入到了模型配置内部，模型据此才能长出正确的“动作神经”。同时，程序还会根据你传入的 `fp16/bf16` 参数推导出训练使用的计算精度。

## Chapter 3. 任务配置装载：数据集路径和相机配置从哪里来

这一章讲的是程序怎么知道“训练哪个任务”。

任务配置都放在 [aloha_scripts/constants.py:6-19](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/aloha_scripts/constants.py#L6-L19) 的 `TASK_CONFIGS` 里。每个任务至少定义三件事：

- `dataset_dir`：数据集所在目录
- `episode_len`：轨迹最大长度
- `camera_names`：训练时启用的相机视角名字

在 [train_tinyvla.py:242-255](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L242-L255) 中，`main()` 会根据 `config['data_args'].task_name` 去 `TASK_CONFIGS` 查表，并取出当前任务的：

- `dataset_dir`
- `episode_len`
- `camera_names`
- `stats_dir`
- `sample_weights`
- `train_ratio`
- `name_filter`

也就是说，训练任务名是一个“入口标签”，而真正控制数据如何被读取的是 `TASK_CONFIGS` 里的细节配置。

**关键代码位置**

- [aloha_scripts/constants.py:6-19](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/aloha_scripts/constants.py#L6-L19)
- [train_tinyvla.py:242-255](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L242-L255)


## Chapter 4. 模型搭建：LLaVA-Pythia 主干如何被加载

这一章讲的是训练开始前，模型是怎样被搭建起来的。

在 [train_tinyvla.py:257-267](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L257-L267) 中，程序先根据 `model_name_or_path` 加载 tokenizer，然后把 `config` 和 `llava_pythia_config` 交给 [llava-pythia/llava_pythia/llava_pythia_utils.py:74-292](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/llava_pythia_utils.py#L74-L292) 的 `load_llava_pythia()`。

- **`load_llava_pythia()` 负责的是模型装配、权重加载、冻结配置、LoRA 注入、视觉处理器设置**
- **真正的动作头结构定义本身，是在 [llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py:35-74](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py#L35-L74) 的 `LlavaPythiaForCausalLM.__init__()` 中完成的**

也就是说，这一章可以分成两层看：

1. `load_llava_pythia()` 把“训练所需的模型系统”组装起来
2. `LlavaPythiaForCausalLM` 自己定义“这个系统内部长什么样”

在 `__init__()` 里可以看到：

- 若 `action_head_type == 'act'`，就构造 ACT 风格动作头
- 若 `action_head_type == 'droid_diffusion'`，就构造 diffusion 动作头

**关键代码位置**

- [train_tinyvla.py:257-267](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L257-L267)
- [llava-pythia/llava_pythia/llava_pythia_utils.py:74-292](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/llava_pythia_utils.py#L74-L292)
- [llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py:35-74](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py#L35-L74)

## Chapter 5. 参数训练策略：冻结、LoRA、可训练模块怎么安排

这一章讲训练时到底哪些参数在更新。

关键逻辑在 [llava-pythia/llava_pythia/llava_pythia_utils.py:173-277](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/llava_pythia_utils.py#L173-L277)。这里依次做了：

1. `freeze_backbone`：是否冻结语言主干
2. `freeze_vision_tower`：是否冻结视觉编码器
3. `gradient_checkpointing`：是否开启梯度检查点
4. `lora_enable`：是否注入 LoRA
5. `tune_mm_mlp_adapter`：是否训练多模态投影层
6. 强制让 `embed_out` 和 `proj_to_action` 保持可训练

这说明 TinyVLA 的训练策略并不是“全模型一起学”，而是更像“在冻结的大脑上，微调视觉接头、多模态桥接层和动作输出层”。

`scripts/train.sh` 的默认配置也印证了这一点：  
在 [scripts/train.sh:21-38](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/scripts/train.sh#L21-L38) 中，LoRA 被开启，视觉塔和 backbone 被冻结，多模态投影层被允许训练。

**关键代码位置**

- [llava-pythia/llava_pythia/llava_pythia_utils.py:173-188](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/llava_pythia_utils.py#L173-L188)
- [llava-pythia/llava_pythia/llava_pythia_utils.py:190-205](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/llava_pythia_utils.py#L190-L205)
- [llava-pythia/llava_pythia/llava_pythia_utils.py:206-228](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/llava_pythia_utils.py#L206-L228)
- [llava-pythia/llava_pythia/llava_pythia_utils.py:230-240](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/llava_pythia_utils.py#L230-L240)
- [scripts/train.sh:21-38](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/scripts/train.sh#L21-L38)

## Chapter 6. 数据发现与切分：训练集和验证集如何划分

这一章讲数据是怎么被发现并拆成 train/val 的。

核心函数是 [data_utils/datasets.py:394-463](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L394-L463) 的 `load_data()`。它内部会：

1. 调用 [data_utils/datasets.py:372-381](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L372-L381) `find_all_hdf5()` 递归查找所有 `.hdf5`
2. 根据 `name_filter` 过滤文件
3. 根据 `train_ratio` 在第一个数据目录上做 episode 划分
4. 构造 train 和 val 的 episode id 列表
5. 最后创建 `EpisodicDataset`

这里的切分单位不是单帧，而是 **episode**。也就是说，训练和验证不是随机拆帧，而是先拆轨迹，再从轨迹里采样时间步。这对机器人学习很重要，因为同一条轨迹里的帧之间高度相关。

**关键代码位置**

- [data_utils/datasets.py:372-381](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L372-L381)
- [data_utils/datasets.py:394-463](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L394-L463)

疑问：训练集和数据集的理解：
- **训练集**：给 AI 看的几万条机器人抓东西的录像（包含图片、语音指令和**人类遥控的真实坐标**）。AI 用它来调整自己的神经网络参数。
    
- **验证集**：留出几百条录像，AI 训练时**绝对不能看**。每训练一段时间，就让 AI 在验证集上“做题”，对比它预测的坐标和人类的真实坐标，看看它是不是真学会了。
疑问2：数据集都是一般都是通过摇操作收集到的数据对吧，那每一个数据集都应该是完整的一段视频。那如果把视频切分了切成每一张图片，那要怎么训练。你指令就是一段文字，但是你是根据这个文字做出一连串的动作
VLA模型一般用的都是动作分块策略。也就是让模型看当前的一张照片，然后预测出未来几步的路径。训练的时候也是这样，每一段摇操作的视频，切成每一个画面，通过给模型看当前画面来让模型去预测未来几步。

# Chapter 7. 统计量计算：为什么训练前要算归一化参数
理解“归一化参数（Normalization Stats）”的作用，就等于解开了你之前一直卡在 `dataset_stats.pkl` 找不到的终极谜团。

你可以把“归一化参数”想象成一个“翻译官的密码本”。

### 1. 为什么“训练前”必须先算这个参数？

因为这个密码本必须是**全局统揽**的。

- 在训练正式开始前，程序（`get_norm_stats` 函数）必须把成百上千个 HDF5 录像文件全部“扫”一遍。
    
- 它的目的是统计出所有录像中，机械臂的 `action`（动作）和 `qpos`（状态）的全局极值和平均情况：也就是找出动作的**最大值（action_max）、最小值（action_min）、平均值（action_mean）和标准差（action_std）**。
    
- 如果你不提前看完所有数据，你就不知道机械臂的极限在哪里（比如它最多能伸多远，最少能缩多短）。
    

### 2. 这个参数到底是干什么用的？（核心作用）

根据 Chapter 7 的分析，这个密码本有两大核心作用：

#### 作用一：在“训练时”，把真实数据变成 AI 喜欢的样子（正向翻译）

- **痛点**：在真实的物理世界里，机械臂的数据差异极其巨大。比如“夹爪开合度”可能是 0 到 1 之间的小数，而“机械臂移动距离”可能是 -500 到 500 的大数字。神经网络非常讨厌这种大跨度、尺度不一的数字，直接喂给它会导致模型“发疯”，梯子爆炸，根本学不会。
    
- **解决**：归一化参数会在训练时，把这些乱七八糟的真实数值，全部**映射（压缩）到更适合神经网络学习的数值范围**（通常在 -1 到 1 之间，或者在 0 附近）。
    

#### 作用二：在“推理/部署时”，把 AI 的输出变回真实动作（反向翻译）

- **痛点**：当你在云服务器上跑 `eval_real_franka_6.py` 测试时，AI 大脑经过一通计算，输出的也是类似 `0.35` 这种在 -1 到 1 之间的高度压缩的抽象数字。机械臂收到 `0.35` 是懵的：我是该转 0.35 度，还是移动 0.35 米？
    
- **解决**：这时就需要用到你那个缺失的 `.pkl` 文件！程序会用它**把模型输出的归一化动作，反变换回真实动作空间**（比如算出来其实是要求移动 25 厘米），然后机械臂才能真正执行。
### 这一章讲的是归一化统计量的来源。

在 [data_utils/datasets.py:309-370](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L309-L370) 的 `get_norm_stats()` 里，代码会遍历所有 episode，读取：

- `/action`
- `/observations/qpos`
- `/observations/qvel`

然后计算：

- `action_mean`
- `action_std`
- `action_min`
- `action_max`
- `qpos_mean`
- `qpos_std`

这些统计量后面会直接用于训练时的动作/状态归一化，也会在推理时用于动作反归一化。所以它不是训练前的附带小工具，而是训练闭环的一部分。

尤其要注意：

- ACT 分支使用 mean/std 标准化
- diffusion 分支使用 min/max 映射到 `[-1, 1]`

所以这一步本质上是在为不同动作头准备不同的数据尺度。

**关键代码位置**

- [data_utils/datasets.py:309-370](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L309-L370)
- [data_utils/datasets.py:441-450](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L441-L450)

## Chapter 7. 统计量计算：为什么训练前要算归一化参数

这一章讲的是归一化统计量的来源。

在 [data_utils/datasets.py:309-370](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L309-L370) 的 `get_norm_stats()` 里，代码会遍历所有 episode，读取：

- `/action`
- `/observations/qpos`
- `/observations/qvel`

然后计算：

- `action_mean`
- `action_std`
- `action_min`
- `action_max`
- `qpos_mean`
- `qpos_std`

这些统计量后面会直接用于训练时的动作/状态归一化，也会在推理时用于动作反归一化。所以它不是训练前的附带小工具，而是训练闭环的一部分。

尤其要注意：

- ACT 分支使用 mean/std 标准化
- diffusion 分支使用 min/max 映射到 `[-1, 1]`

所以这一步本质上是在为不同动作头准备不同的数据尺度。

**关键代码位置**

- [data_utils/datasets.py:309-370](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L309-L370)
- [data_utils/datasets.py:441-450](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L441-L450)

## Chapter 8. 单样本生成：一条 episode 数据如何变成训练样本
这一章是训练数据流最关键的一章之一。

单样本的核心逻辑在 [data_utils/datasets.py:85-200](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L85-L200) 的 `EpisodicDataset.__getitem__()`。这一步会做很多事情：

1. 通过 `_locate_transition()` 把全局索引映射到某个 episode 和某个时间步
2. 从 HDF5 中读取：
    - 当前时间步图像
    - 当前 `qpos`
    - 全部后续 `action`
    - `language_raw`
3. 根据当前时刻构造动作 chunk
4. 用 `is_pad` 标记超出轨迹长度的无效动作位
5. 把多相机图像堆起来
6. 做 resize、RGB/BGR 处理、归一化
7. 根据动作头类型做 action normalization
8. 对状态做 `qpos` 标准化
9. 最后把样本交给 `llava_pythia_process.forward_process()`

这一章要强调：TinyVLA 的一个“训练样本”不是单独的一张图或单步动作，而是“当前观测 + 当前语言 + 当前状态 + 未来动作片段”。

**关键代码位置**

- [data_utils/datasets.py:77-82](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L77-L82)
- [data_utils/datasets.py:85-200](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L85-L200)

## Chapter 9. 多模态预处理：图像、语言、状态怎么翻译成模型输入

这一章讲的是样本如何从机器人格式转成 LLaVA 能吃的格式。

核心类是 [data_utils/datasets.py:204-307](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L204-L307) 的 `LlavaPythiaProcess`。

这里主要做三件事：

1. `datastruct_droid2llava()`  
    把原始样本包装成 LLaVA 风格的对话格式，默认模板是：
    
    - human: `"<image>\n" + raw_lang`
    - gpt: `" "`
2. `parse_image()`  
    把多视角图像做 pad / preprocess，交给 image processor
    
3. `forward_process()`  
    调用 `preprocess_multimodal()` 和 `preprocess()` 生成：
    
    - `input_ids`
    - `labels`
    - `image`
    - `image_r`
    - `image_top`
    - `state`
    - `action`
    - `is_pad`

这一章本质上是“把机器人监督学习样本翻译成 LLaVA 多模态监督学习样本”。

**关键代码位置**

- [data_utils/datasets.py:204-307](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/datasets.py#L204-L307)
### 第一步：处理图片

大模型的“视网膜”（视觉编码器，比如 CLIP）通常只接受固定大小的**正方形**图片（比如 336x336）。但是你机械臂摄像头拍出来的画面往往是长方形的（比如 180x320）。

- 在 `parse_image()` 中，程序会根据配置（`image_aspect_ratio`）来处理图片。
    
- 如果设置为 `pad`，它不会硬生生把长方形压扁成正方形，而是在长方形的上下（或左右）**补上黑边**，凑成一个正方形，以保证画面不变形。
    

### 第二步：伪造聊天记录（最核心的欺骗）

这是整个处理中最有趣的一步。LLaVA 是一个**视觉问答模型**，它习惯的输入模式是：“人类发一张图问问题 -> GPT回答”。

- 但在机器人任务里，哪有聊天啊？只有一句干巴巴的指令（`raw_lang`），比如“抓住苹果”。
    
- 在 `datastruct_droid2llava()` 函数中，代码强行写了一个“对话剧本”：
    
    - **假装是人类 (human) 发的**：`"<image>\n" + "抓住苹果"`。
        
    - **假装是 GPT 准备回答**：`" "`（留空，等模型输出动作）。
        
- **意义**：通过这种方式，它成功骗过了 LLaVA 主干，让 LLaVA 以为自己又在做一次普通的“看图说话”任务，而实际上它是在做“看图出动作”的任务。
    

### 第三步：装箱打包发货

把上面处理好的图片、伪造好的“聊天记录”，还有机器人的关节状态（State）和真实动作（Action）统统塞进一个字典（Dict）里。

- 在这个叫 `forward_process()` 的函数里，它会把那段假聊天记录变成数字编号（`input_ids`）。
    
- 最后输出一个标准包裹给 Chapter 12 的前向传播去使用：里面包含 `input_ids`（文本）、`image`（图片）、`state`（状态）、`action`（动作目标）。

## Chapter 10. Batch 组装：多个样本如何拼成一个训练 batch

这一章讲 collator。**Chapter 8 (单样本生成)**：是流水线的**起点**，负责从巨大的 HDF5 文件中“扣”出某一个时刻的数据片段，**Chapter 9 (多模态预处理)**：是流水线的**转换站**，负责把 Chapter 8 提取出的“理科生数据”（坐标、关节值）翻译成大模型能听懂的“对话格式”，而**Chapter 10 (Batch 组装)**：是流水线的**终点**，负责把多个经过 Chapter 9 转换后的样本整整齐齐地“捆绑”在一起，交给训练器。

在 [data_utils/processor.py:496-566](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/processor.py#L496-L566) 的 `DataCollatorForSupervisedDataset` 中，代码把多个样本合并成一个 batch。主要操作包括：

- 对 `input_ids` 做 pad
- 对 `labels` 做 pad，并用 `IGNORE_INDEX` 填充
- 生成 `attention_mask`
- 堆叠 `actions`
- 堆叠 `states`
- 堆叠 `images/images_r/images_top`
- 堆叠 `is_pad`

这里的关键点是：这个 collator 不只是处理文本，还要同时处理图像和动作监督，所以它是一个真正的多模态 batch 组装器。

没有这一层，Trainer 只知道怎么处理 NLP 样本，不知道怎么同时喂视觉和动作张量。

**关键代码位置**

- [data_utils/processor.py:496-566](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/data_utils/processor.py#L496-L566)

## Chapter 11. 训练器启动：Trainer、采样器、优化器怎么协同工作

这一章讲训练基础设施。

在 [train_tinyvla.py:192-204](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L192-L204) 中，`train_bc()` 会创建 `LLaVAPythiaTrainer`。这个 trainer 的实现位于 [llava-pythia/llava_pythia/train/llava_pythia_trainer.py:168-347](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/train/llava_pythia_trainer.py#L168-L347)。

它主要做三件事：

1. `get_train_dataloader()`  
    使用自定义 `CustomBatchSampler` 来按 episode 长度和采样权重取训练 batch
    
2. `get_eval_dataloader()`  
    构造验证集 dataloader
    
3. `create_optimizer()`  
    按参数类别分组，给 LoRA 参数和非 LoRA 参数设置不同学习率，尤其会把 `mm_projector`、`embed_out`、`proj_to_action` 等归到特殊组
    

这一章的重点不是“Trainer 会训练”，而是说明 TinyVLA 对采样和优化器都做了定制，不是 Hugging Face Trainer 的原样使用。

**关键代码位置**

- [train_tinyvla.py:192-204](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L192-L204)
- [llava-pythia/llava_pythia/train/llava_pythia_trainer.py:168-347](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/train/llava_pythia_trainer.py#L168-L347)
### 结合代码深入理解 Chapter 11

在 `llava_pythia_trainer.py` 中，我们可以清晰地看到这三个核心组件是如何被定义和协同的：

**1. 采样器 (Sampler)：挑题目的艺术** 在 `get_train_dataloader` 函数中，代码没有使用 PyTorch 默认的顺序采样，而是使用了自定义的 `CustomBatchSampler`。 这个 `CustomBatchSampler` 的作用是：它知道每个录像（Episode）有多长，它会根据你设定的概率，每次随机挑出几个时刻的数据，拼成一个 Batch 交给模型去学。这就好比老师每次随堂测验，都是从题库里随机抽几道题，防止模型按照固定顺序死记硬背。

**2. 训练器 (Trainer)：流水线大管家** `LLaVAPythiaTrainer` 继承自 Hugging Face 的标准 Trainer。它是整个训练过程的总指挥。它负责：

- 从 Sampler 那里拿到数据包裹（Batch）。
    
- 把数据送入模型进行前向传播（Forward），计算出动作预测的误差（Loss）。
    
- 调用后向传播（Backward）计算梯度。
    

**3. 优化器 (Optimizer)：因材施教的调参师** 这是这段代码里最精彩的部分。在 `create_optimizer` 函数中，你会看到很长的一段参数分组逻辑 (`optimizer_grouped_parameters`)。 对于像 TinyVLA 这种复杂的机器人大模型，不能所有参数都用同一个速度去学习：

- **语言主干（大脑）**：通常会冻结或者用极小的学习率微调（因为怕它变傻，忘了以前学的通用知识）。
    
- **视觉接头 (`mm_projector`) 和 动作头 (`embed_out`, `proj_to_action`)**：这是新长出来的器官，代码里把它们归类为 `non_lora_parameters`，并给它们分配了一个单独的、通常更大的学习率 (`self.args.non_lora_lr`)。 这就像是在健身房：教练（Optimizer）给你的手臂（动作头）安排大重量训练，而让你的核心（主干模型）只做轻量级的维持性训练。

## Chapter 12. 前向与损失：模型在训练时到底学什么

这一章是整个训练说明书的“核心中的核心”。

主前向在 [llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py:130-223](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py#L130-L223)。

训练时，模型会先做多模态拼接，再得到 `hidden_states`，然后根据 `head_type` 分支：

### 1. ACT 头

在 [llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py:274-311](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py#L274-L311)

它会：

- 用 `proj_to_action` 把语言主干隐藏状态映射到动作头输入空间
- 调用 `embed_out(...)` 预测动作
- 计算动作重建的 `l1 loss`
- 计算潜变量的 `kl divergence`
- 最终总损失为 `l1 + kl * kl_weight`

### 2. Diffusion 头

在 [llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py:314-386](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py#L314-L386)

它会：

- 先给真实动作加噪声
- 随机采样 diffusion timestep
- 用 `ConditionalUnet1D` 预测噪声
- 用 MSE 约束预测噪声和真实噪声一致
- 用 `is_pad` 屏蔽无效动作位

这一章要讲清楚：训练目标不是“生成文本”，而是借助多模态 VLM 主干学到一个可控的动作表示，最后通过动作头把 hidden states 拉到动作空间。

**关键代码位置**

- [llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py:130-223](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py#L130-L223)
- [llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py:274-311](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py#L274-L311)
- [llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py:314-386](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py#L314-L386)
### 理解：
你可以把前面的章节看作是为模型准备“教材”和“身体”，而 Chapter 12 则是模型真正的“学习”（前向传播）过程。

在 `llava_pythia.py` 这段代码中，最核心的方法就是 `forward()`。它接过了 Chapter 10 打包好的那些批次数据（图片、文本、动作），并让它们在模型内部流转。

这个“流转”过程可以分为四个清晰的步骤：

### Step 1: 视觉与文本的融合 (多模态输入准备)

在 `forward()` 函数的开头，第一件事是调用 `prepare_inputs_labels_for_multimodal`。 大模型的“语言主干”原本只能处理文字（Token）。这一步的作用是，把机器人的图片（经过视觉编码器处理后的特征）像“插队”一样，塞进原本的文字序列里。这样，语言模型就能把“看到的东西”和“听到的指令”结合在一起理解了。

### Step 2: 语言主干的“沉思” (生成 Hidden States)

融合后的数据被送入了真正的“大脑”——GPTNeoX 主干（`self.get_model()`）。 经过一层层复杂的神经网络计算，大脑输出了一堆高维度的数字，这就是代码里的 `hidden_states`。 你可以把 `hidden_states` 想象成模型在说：**“根据现在的场景和我收到的指令，我已经理解了当前的局势。”** 这是非常抽象的内部思考结果。

### Step 3: 根据“动作头”分道扬镳

大模型的“思考结果”（`hidden_states`）不能直接控制机械臂。它需要一个“翻译官”或“小脑”，把它变成具体的机械臂坐标。这个东西在代码里叫 **Action Head (动作头)**。 在 `forward()` 中，代码会根据你在配置中设定的 `head_type`，进入不同的处理分支：

- 如果是普通的全连接层：走 `forward_fc_head`。
    
- 如果是 ACT 模型风格：走 `forward_act_head`。
    
- 如果是基于扩散模型（Diffusion）：走 `forward_diffusion_head`。
    

### Step 4: 不同的“小脑”，不同的学习目标

这里是最难，但也最精妙的地方。TinyVLA 支持两种主流的动作头，它们的“学习方式”完全不同：

#### A. ACT 动作头 (学习直接重建动作)

- 如果走 `forward_act_head` 分支，模型会把 `hidden_states` 投影成适合动作头的格式，然后直接尝试“默写”出未来一段（比如 16 步）的动作。
    
- **如何算对错（Loss）？**
    
    1. **L1 重建误差**：把模型默写出来的动作（`a_hat`），和数据集中人类真实的动作做对比，看差了多少。
        
    2. **KL 散度**：ACT 动作头内部带有一个 VAE（变分自编码器）结构，这部分是要求模型在输出动作的同时，保持内部对动作分布的理解是规整的（避免胡乱动作）。
        

#### B. Diffusion (扩散) 动作头 (学习如何从噪音中恢复动作)

- 如果走 `forward_diffusion_head` 分支，它的思路非常清奇： 它不是直接让模型预测动作。相反，它先把一段人类真实的动作“搞砸”——往里面加入随机噪音，变成一堆乱七八糟的 `noisy_actions`。 然后，它把这些乱数据，连同大脑的思考结果（`hidden_states`），一起交给扩散模型（`ConditionalUnet1D`）。
    
- **学习目标是什么？** 模型不需要直接输出动作，而是要**预测刚才被加进去的“噪音”是什么** (`noise_pred`)。
    
- **如何算对错（Loss）？** 计算预测的噪音和真实加进去的噪音的差距（MSE Loss）。只要模型学会了识别噪音，在实际部署（推理）时，它就能从一团纯噪音中，一步步把正确的动作“雕刻”出来。

## Chapter 13. 收尾与产物：checkpoint、LoRA 权重和统计文件怎么保存

这一章讲训练结束后留下什么。

在 [train_tinyvla.py:204-224](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L204-L224) 中，`train_bc()` 结束后会先 `trainer.save_state()`，然后根据是否启用 LoRA 分成两条保存路径：

- 如果 `lora_enable=True`  
    调用 [llava-pythia/llava_pythia/llava_pythia_utils.py:320-386](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/llava_pythia_utils.py#L320-L386) 提取：
    
    - LoRA 参数
    - 非 LoRA 可训练参数  
        然后保存：
    - adapter 权重
    - `non_lora_trainables.bin`
- 如果 `lora_enable=False`  
    走 [llava-pythia/llava_pythia/llava_pythia_utils.py:388-412](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/llava_pythia_utils.py#L388-L412) 的安全保存逻辑
    

此外在 [train_tinyvla.py:281-284](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L281-L284)，还会把数据归一化统计量保存成 `dataset_stats.pkl`。这个文件非常关键，因为推理时动作反归一化就靠它。

再往后，仓库还提供了 [scripts/process_ckpts.sh](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/scripts/process_ckpts.sh) 做 checkpoint 后处理，把训练产物整理成更适合部署/推理使用的权重格式。

**关键代码位置**

- [train_tinyvla.py:204-224](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L204-L224)
- [train_tinyvla.py:281-284](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/train_tinyvla.py#L281-L284)
- [llava-pythia/llava_pythia/llava_pythia_utils.py:320-412](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/llava-pythia/llava_pythia/llava_pythia_utils.py#L320-L412)
- [scripts/process_ckpts.sh](vscode-webview://0p9qsjrqjii0p316tgsqo4m2r2m3br8kpcf92h05rosdohh94sf3/scripts/process_ckpts.sh)