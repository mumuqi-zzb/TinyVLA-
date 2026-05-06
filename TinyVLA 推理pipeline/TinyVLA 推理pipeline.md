# Chapter 1：一些初始化的配置
推理的入口是在 : **eval_real_franka.py**
程序一启动，先不是急着看图、也不是急着出动作，而是先把“主角档案”准备好：模型路径、动作头类型、语言指令、图像尺寸、环境句柄。这一步的意义很简单——先把整场表演的剧本、舞台和演员都摆好，后面的每一步才知道自己该做什么。
主要是在这个文件里面的main函数里面。

## ⚙️ 核心动作 (How)

1. 入口落在 `eval_real_franka.py` 的 `__main__`。
2. 这里先定义 `policy_config`，包括 `model_path`、`model_base`、`enable_lora`、`conv_mode`、`action_head`。
3. 再定义 `raw_lang`，这是本次任务的自然语言指令。
4. `llava_pythia_act_policy(policy_config)` 负责加载 tokenizer、VLM 主干、图像处理器、配置对象。
5. 最后把 `policy` 和 `deploy_env` 交给 `eval_bc(...)`，真正进入推理循环。


### 1.配置 policy_config 
在这个里面，model path是就是训练好的微调权重（也就是LORA权重）要保存在哪里，然后model base是原始的大模型也就是基座模型。这里配置就是要把这两个保存在哪里。
然后enable lora表示是否启用Lora低秩微调（这是一个开关。如果设为 True，程序就会执行上面提到的“合并”操作；如果设为 False，程序就会认为 model_path 里存的是一个完整的模型，直接整个加载)
conv mode:全称conversation model。首先需要知道的是，外界的图像和指令最终会转变成向量传给模型，模型处理这些向量，然后生成文本。这个模型就类似于平常用的AI，给他问题，他吐出答案，只不过这里吐出的答案不是最终的动作，而是融合全部信息的特征向量，特征向量要给小脑，让小脑吐出动作。再回来，这里conv mode就是规定我们要给模型的内容的格式是什么样的，只有按照他的格式，它才知道这个是啥意思，我要做啥。
最后一个action head，这里配置是因为在VLA的代码中，作者通常会在代码里准备好几种不同的算法模块，方便做对比实验。这个代码里面action head有三种:fc（全连接层），act（类似VAE的架构），droid diffusion（扩散模型）。
### 2.配置raw_lang，也就是自然语言指令
### 3.配置 policy，选择llava_pythia_act_policy
这里面主要是配置三个部分: 
tokenizer（文字翻译官，加载分词器，负责把语言指令变成模型能懂的数字ID）
image_processor（图像处理器，负责把摄像头拍到的画面裁剪，补齐）
policy（代码里面是load_policy，在这里面配置了三个部分，也就是前面说的三个，lora部分，基座部分，小脑部分。前面mode path，mode base只是告诉模型这里权重位置在哪里，相当于给了一张清单，此时模型还没有被加载，显存还是空的。这里load policy 就是接受上面的清单作为输入参数，去硬盘里面搬运模型，把数据塞进显卡里面）

### 4.最后把policy和deploy_env交给eval_bc ，由此进入推理的循环。
# Chapter 2: 模型登场：把多模态大脑装进 GPU
这一步是在给模型“通电”。语言模型、视觉塔、投影层、动作头都要一起加载，不然它只能看文字，不能看图，也不能直接变成机械臂动作。目标是把一个会“理解图像+文字+状态”的控制器装进显存。

## ⚙️ 核心动作 (How)

1. `load_pretrained_model(...)` 根据 `model_name` 判断是否走 Pythia 分支。
2. 如果是 LoRA 模型，就先加载 base model，再合并 LoRA 权重。
3. 视觉处理器根据 vision tower 类型选择 `CLIPImageProcessor` 或 `SiglipImageProcessor`。
4. tokenizer 会补上图像 token，保证文本里能插入图片占位符。
5. `LlavaPythiaForCausalLM` 最终被送到 CUDA（指的是`LlavaPythiaForCausalLM`的**实例**以及这个实例携带的**海量权重数据**。），并切到推理可用状态。

首先先讲一下一些理解的误区:在tinyvla这个项目中，采用的是LLaVA的架构， LLaVA 是一个经典VLA架构，用 CLIP 视觉编码器 + LLM（语言大模型） 做端到端多模态对齐，属于标准的 VLM。然后这里选用的LLM就是pythia，tinyvla并没有训练整个pythia，而是把模型分成了两个部分，分别是基座模型（model_base）和lora微调权重（model_path），只训练lora的部分。整体流程就是，外界把图像传给视觉编码器转成特征向量，特征向量再经过投影层转成和文字向量一样的维度格式，然后传给pythia模型里面就可以进行推理了。这个过程就是VLM

而在第二个流程目的就是把让模型学会看图，在第一章节只配置了很基础的操作，没有涉及到图像编码器。
首先是进入load_pretrained_model（），先根据model_name判断当前是不是pythia的模型。根据是或不是做对应的操作。如果是pythia，那就会进行融合的操作，把基座模型和lora微调权重合成在一起（这里方法看论文笔记。合成后的pythia就变成了一个能懂得物理抓取的模型）

接着，配置视觉编码器，也就是配全VLM架构。代码会根据“vision tower”判断选择CLIP还是Siglip。

然后是优化分词器，这里代码会强制往分词器的字典里塞入特殊的占位符。这样，当文本指令里面突然出现一张图片，tokenizer就能够理解这里放的是图片。

然后是在llava_pythia.py文件里面 ，在LlavapythiaForCausalLM里面，安装小脑。根据所选择的配置，去创建对应的动作生成模块。如果选择ACT，它会建立一个基于transformer的动作头，如果选择Diffution，它会准备好去噪调度器和U-Net网络。

然后是在llava_arch.py文件里面，这里会根据我们前面在load_pretrained_model（）所确定的视觉编码器，对要用的视觉编码器进行具体配置。这一步会把选定的视觉塔（Vision Tower）实例化，并准备好加载它的预训练权重。以及配置投影层

# Chapter 3. 观测采集：机械臂开始“睁眼”
真正的推理不是只盯着一张图，而是要同时拿到“眼睛看到什么”和“身体现在是什么状态”。图像告诉模型外部世界，机器人状态告诉模型自己此刻站在哪里、姿态怎样。只有两者合在一起，动作才不会像盲人摸象。
### ⚙️ 核心动作 (How)

1. 在循环里先调用 `deploy_env.get_observation()`。
2. 观测交给 `get_obs(obs, stats)`，期望返回 `traj_rgb_np` 和 `robot_state`。
3. `traj_rgb_np` 是原始图像数组，通常还是 `uint8`、值域 `0~255`。
4. `robot_state` 是机器人本体状态，随后转成 CUDA 上的 `float32` 张量。
5. 这些数据会在后续被送进 `process_batch_to_llava(...)`。

前面提到的，llavapythiaForCausalLM最终被送到CUDA，并切到推理可用状态。而这里就进入了推理可用状态，即模型进入实战演练。
with.torch.inference_mode()就是表示接下来的计算，不记录梯度，也就表示模型把全部算力用来向前推进运算，不仅速度变快而且节省内存。
DT = 1 / FPS：这定义了机器人的“反应节拍”。比如 FPS 是 10，那么每 0.1 秒，机器人就要完成一次“看图 -> 思考 -> 动关节”的完整闭环。

for t in range(max_timesteps):：这就是那个神圣的死循环。t 代表当前是第几个时间步。每一次循环，机械臂都会往前挪动一点点。

然后执行obs = deploy_env.get_observation()
这就是机器人“睁眼”和“感知身体”的瞬间。
deploy_env 是控制真实机器人的接口。调用这个函数，就像是去读取 Franka 机械臂底层传感器的快照。返回的 obs 是一个极其原始的数据包，里面可能包含了多个摄像头的彩色画面，以及当前机械臂各关节的旋转角度（qpos）、夹爪的开合程度等信息。
拿到原始数据包后，对数据包进行拆解。traj_rgb_np, robot_state = get_obs(obs, stats)
拿到原始数据包 obs 后，代码把它扔进了 get_obs 提取函数里，分成了两股数据流：

traj_rgb_np (视觉流)：这是从摄像头里剥离出来的纯图像数组。此时的它还是非常粗糙的 uint8 格式（像素值在 0~255 之间）。它现在就是一张普通的 Numpy 图片，还没准备好送给神经网络。
robot_state (本体感受流)：这是机器人当前的姿态状态。
然后进行robot_state = torch.from_numpy(robot_state).float().cuda()
这一步是先把机器人状态数据从numpy数组转换成pytorch张量，这样才能在cuda里面运行。.float表示数据类型转换成float32位的，
.cuda是把数据从CPU上搬到GPU上(机械臂的传感器连接在电脑主机，而负责接受这些信号的是CPU和主板)

# Chapter 4. 图像与状态预处理：把原始世界翻译成模型语言
原始图像对模型来说太粗糙，必须先缩放、归一化、补方、拆视角。状态向量也不能直接乱丢，要先转成模型能吞进去的张量。这个阶段就像把现场录像和传感器读数翻译成一套统一的机器语言。

在chapter3里面，我们拿到了机器人的图像和状态数据，再这一章主要是对这两个数据进行处理，把他们转换成模型能够理解语言，也就是张量。
首先进入由chapter3可知，模型进入循环，程序根据时间步看图像，执行
curr_image = torch.from_numpy(traj_rgb_np / 255.0).float().cuda()
原始图像对像素都是255，直接扔给神经网络会引发梯度爆炸，这里除以255把所有像素压缩成0-1之间的浮点数区别，也就是归一化操作。然后用torch.from_numpy.........cuda把numpy数组转换成pytorch要求的单精度张量，并送入显存 准备接下来的精加工。
如果配置了 rand_crop_resize=True，图像会进行一次中心裁剪和尺寸重置，也就是把图像四周5％的边缘砍掉，将剩下的中心区域拉伸回原来的尺寸，这个有利于增强图像在微小变化下的鲁棒性。
然后经过初步清洗的curr_image和刚刚放进GPU的state一起进入process_batch_to_llava 这个专门的“预处理车间”。在这个函数里面，图像先被拆解成双视角，然后再对图像补方（expand2square）。最后把图像交给我们前面配置的视觉编码器，它执行了模型预训练时要求的最标准化的处理（比如减去特定的均值，除以特定的标准差），生成了最终的 image_tensor。
。
经过 `image_processor` 处理后，虽然图像变成了规范的张量，但它通常默认是 `float32` 精度，并且可能还在 CPU 上。 代码执行了：

```
image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)
image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)
```

- **统一设备 (`policy.device`)**：确保图像张量和 LLM 大脑在同一张显卡上（避免跨设备报错）。
    
- **统一精度 (`policy.dtype`)，通常是 fp16**：大语言模型（如 Pythia）为了节省显存和加快推理，通常是用**半精度浮点数 (fp16)** 或 **bfloat16** 加载的。如果用 `float32` 的图像去撞击 `fp16` 的模型，PyTorch 会立刻报 `RuntimeError: expected scalar type Half but found Float`。这一步就是强行把图像精度压缩，与大脑完全对齐。
同时，state也进行同样的操作
states = robo_state.to(self.policy.device, dtype=self.policy.dtype)

# Chapter 5. 语言提示词：模型知道自己在做什么任务
图像和状态只告诉模型“现在是什么”，语言指令告诉它“应该做什么”。这一步把任务目标写进上下文里，让模型不是胡乱猜动作，而是朝着指定目的去行动。
### ⚙️ 核心动作 (How)

1. `raw_lang` 作为原始任务描述被传入 `process_batch_to_llava(...)`。
2. `conv_templates[self.policy_config['conv_mode']]` 生成对话模板。
3. 根据 `mm_use_im_start_end` 决定是否插入图像起止 token。
4. 图像 token 被拼到文本最前面，后面接自然语言任务。
5. `conv.get_prompt()` 生成最终 prompt，再追加 `<|endoftext|>`。
6. `tokenizer_image_token(...)` 把 prompt 变成 `input_ids`。
7. `attention_mask` 随之生成，标记哪些 token 需要参与注意力计算。

- **（拼装原始字符串）** ==把你的自然语言 `raw_lang` ("put the tennis...") 拿过来，在它前面硬塞入几个特殊单词：`<im_start> <image> <im_end>`。这就好比在文章开头画了一个框，告诉模型：“待会儿这儿要放一张图”。==
    
- **（套上对话模板）** 把刚才拼好的句子，塞进 Pythia 专属的“聊天格式”里（比如加上 "User:" 和 "Assistant:" 的前缀）。这让模型进入“听话干活”的角色扮演状态。
    
- **（翻译成数字 ID）**： `tokenizer_image_token` 出马，把这串人类能看懂的字符，彻底翻译成模型脑子里的字典编号（`input_ids`）。注意，那个 `<image>` 单词会被专门翻译成一个特殊的 ID（比如 `IMAGE_TOKEN_INDEX` 可能是 20000）。
    
- **（attention mask）**： 大模型（像 Pythia、GPT）通常是按“批次（Batch）”处理数据的，这就要求每一句话的长度必须完全一样长。 如果你的指令是 `put the tennis ball`（比较短），别人的指令是 `put the red tennis ball into the blue bucket`（比较长），为了凑齐长度，模型会在短句子的末尾塞入一堆无意义的“占位废话”（也就是 Pad Token）。这行代码的作用是：**给大模型发一份“注意力黑名单”（屏蔽掉这些填充的占位符）。**

# Chapter 6. 多模态融合：图像特征和文字在脑内会合

在 Chapter 5 里，我们已经把文本指令变成了一串 ID（比如 `[ID_put, ID_the, ID_image, ID_ball]`），并在里面挖好了一个名为 `<image>` 的坑。

Chapter 6 的终极目标用四个字概括就是：**“填坑合体”**。它要把 Chapter 4 洗好的“真实图像特征”，精准地镶嵌到 Chapter 5 挖好的那个坑里。

这段代码的核心函数是 `prepare_inputs_labels_for_multimodal`。我们可以把它拆解为 4 条极其清晰的流水线操作：

## 1. 榨取图像精华

真正的图像像素目前还是不能用的，必须先提取特征。

- 代码会调用 `self.encode_images`（或者 `get_image_fusion_embedding`）。
    
- 图像张量会被送进你的 Vision Tower（也就是 CLIP 或 SigLIP 眼睛），然后再穿过 `mm_projector`（翻译官）。注意注意：可能会疑惑在chapter4里面已经调用了`image_processor`，变成了 `image_tensor`。这里为什么还要进行处理？这里Vision Tower 并不是 `image_processor`-。image_processor（例如 `CLIPImageProcessor`）只是一个纯数学和像素的转换工具。它只做三件事：调整分辨率大小、减去均值颜色、除以标准差。此时的 `image_tensor` 依然是一张“看得见”的图片（形状类似 `[3, 336, 336]`），只不过它的像素值不再是 0~255，而是变成了标准化后的浮点数（比如 -1.2, 0.5）。
- **它里面还没有任何高级的“语义特征”。**
而当你进入 Chapter 6 的 `prepare_inputs_labels_for_multimodal` 时：

- **第一步：进入 Vision Tower (`encode_images`)**
    
    这里才是真正的神经网络。刚才洗好的 `image_tensor` 被送进 CLIP 或 SigLIP 模型。几百层的 Transformer 结构疯狂运转，把像素彻底碾碎，提取出高级的“视觉特征（Feature）”（比如“这里有个红色的球”的抽象数学表示）。
    
- **第二步：经过 `mm_projector`**
    
    这个翻译官把上面提取出的视觉特征，转换成和文本模型维度完全一样的张量。
    

**总结比喻：** Chapter 4 的 `image_processor` 就像是**洗菜工**，把土豆洗净切块（变成张量）。 Chapter 6 的 `Vision Tower` 才是真正的**大厨**，把土豆块做成最终的红烧肉（视觉特征向量）。

## 2. 定位“坑”的精确坐标

接下来，代码要在 Chapter 5 传过来的一长串文本 ID 中，寻找那个特殊的图片占位符。

- 代码执行了这句：`image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]`。
    
- 这就像是用雷达扫描，精准找出了 `<image>` 这个 token 在句子中的具体索引位置（比如它在第 3 个位置）。

## 3、真正的“移花接木” 
在 Chapter 5 中，我们没有把文本转换成向量，而是只生成了 `input_ids`（一串数字编号）和准备好了对话格式。之所以在 Chapter 6 要把文本分成“图片前”和“图片后”，是因为图像特征不是简单地“挂”在文本末尾，而是要“替换”掉序列中间那个特定的占位符。
为了让你看清代码是如何一步步把视觉特征“嫁接”进文本序列的，我们将代码逻辑拆解为**三个物理阶段**：
#### 1. 寻找切口：定位图像占位符

手术的第一步是找到 `input_ids`（文本 ID 序列）中那个虚假的占位符 `IMAGE_TOKEN_INDEX`。

Python

```
# 代码位置：llava_arch.py
# 遍历 batch 中的每一条数据
for batch_idx, cur_input_ids in enumerate(input_ids):
    # 找到图像标记在输入序列中的具体位置坐标
    image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0] 
```

这里的 `image_token_indices` 就是手术刀要切下去的精准坐标。

#### 2. 分离与植入：真正的“嫁接”过程

这是函数中最核心的 `while` 循环部分。代码就像剪辑师一样，把文本向量切开，再把视觉特征向量缝在中间。

Python

```
while image_token_indices.numel() > 0:
    cur_image_features = image_features[cur_image_idx] # 拿到洗好的视觉特征（真肉身）
    image_token_start = image_token_indices[0]         # 拿到当前占位符的起始坐标

    # 【动作 A：处理图片前的文本】
    # 将占位符之前的 ID 查表转成向量，存入列表
    cur_new_input_embeds.append(self.get_model().embed_in(cur_input_ids[:image_token_start]))

    # 【动作 B：移花接木（植入真身）】
    # 直接在列表末尾追加 576 个视觉特征向量，完全取代原本的占位符 ID
    cur_new_input_embeds.append(cur_image_features)

    # 【动作 C：更新剩余序列】
    # 将指针移到占位符之后，准备处理剩余的文本内容
    cur_input_ids = cur_input_ids[image_token_start+1:]
    
    # 重新搜索剩余序列里是否还有图片占位符
    image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
```

通过这一步，代码成功地将来自 Vision Tower 的**非文本信号**（`cur_image_features`）强行缝合进了语言模型（LLM）的生命线里。

---

#### 3. 缝合收尾：合并与对齐

手术完成后，`cur_new_input_embeds` 列表里存着三部分：图片前的词向量、图片特征、图片后的词向量。最后一步是把它们“粘”起来。

Python

```
# 将列表里的多段张量沿着长度维度拼成一条完整的向量长龙
cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
new_input_embeds.append(cur_new_input_embeds)
```

**为什么这步必不可少？**

- **物理维度统一**：`embed_in` 处理出来的词向量维度（比如 1024）必须和 `mm_projector` 处理出来的视觉向量维度完全一致，否则 `torch.cat` 拼接会直接报错。
    
- **隐藏身份**：拼接完成后，这串向量就被送入了 `GPTNeoX` 主干。对于主干网络来说，它并不知道哪些向量是“原装”的词向量，哪些是“嫁接”过来的视觉特征。它只会一视同仁地对它们进行 Self-Attention 计算。
### 疑问1？为什么要分“图片前”和“图片后”？

想象你在写一条朋友圈，文字是：“今天天气真好 [图片] 咱们去踢球吧。”

- **原始序列**：`[今天, 天气, 真好, <IMAGE_PLACEHOLDER>, 咱们, 去, 踢球, 吧]`。
    
- **替换逻辑**：
    
    1. 你不能直接把整段话变向量，因为模型的大脑字典里根本没有“图片肉身”的条目，它只认识“文字”。
        
    2. 你需要把 `<IMAGE_PLACEHOLDER>` **左边**的文字拿出来，去查字典变向量。
        
    3. 你需要把 `<IMAGE_PLACEHOLDER>` **右边**的文字也拿出来，去查字典变向量。
        
    4. 最后，你把**左向量 + 真实的图片特征向量 + 右向量** 像粘胶带一样粘在一起。
        

**如果不分前后：** 真实的图片特征就会把所有的文字都冲掉，或者只能傻傻地待在最开头或最末尾，无法实现“图文穿插”的灵活排列。

#### 结合代码看“缝合”现场

在 `llava_arch_2.py` 中，这段逻辑体现得非常清晰：

- **处理“图片前”**： `cur_new_input_embeds.append(self.get_model().embed_in(cur_input_ids[:image_token_start]))` 这一行代码调用了 `embed_in`（真正的文本变向量工具），把占位符之前的所有文字 ID 一次性转成了向量。
    
- **植入“图片肉身”**： `cur_new_input_embeds.append(cur_image_features)` 直接把我们在 Chapter 6 第一阶段炼出来的图像特征（来自 Vision Tower 的真家伙）塞进列表。
    
- **处理“图片后”**： `cur_input_ids = cur_input_ids[image_token_start+1:]` 指针跳过那个已经被“榨干”价值的占位符 ID，去处理剩下的文字 ID。随后循环会再次调用 `embed_in` 把剩下的文字也变成向量。
### 疑问：前面chapter的代码写的input_id的顺序是文本在最前面，而不是中间？
代码必须写得“通用”，以防万一

Chapter 6 的 `prepare_inputs_labels_for_multimodal` 函数（位于 `llava_arch.py`）是一个**通用底层函数**。 它不能假设图像一定在开头，因为：

1. **多轮对话**：如果是第二轮对话，图像可能在很靠后的位置。
    
2. **不同模板**：有的模型模板可能要求先说“你好”，再传图片。

## 第四阶段：修补注意力掩码

由于我们在序列中间强行插入了 $576$ 个向量，原本 Chapter 5 计算出的 `attn_mask` 长度对不上了。

- **扩充掩码**：代码会自动在原本的 `attention_mask` 中插入对应长度的 `True`（或 1），确保模型在计算注意力时，眼睛能盯着这 $576$ 个视觉特征看。
    
- **设备与精度**：最后再次确认，这串长达几百甚至上千维的混合向量，全部都在 `cuda` 上，且精度统一为 `fp16`。

# Chapter 7. 主干推理：语言模型开始思考
输入已经准备完毕，接下来轮到语言模型主干做“理解与推理”。它会把多模态输入编码成隐藏状态，相当于在脑海里构建一份当前场景的内部表征。动作头不直接看原始图像，而是看这份高维思考结果。
如果说前面的章节都是在做“外围准备”，那么 Chapter 7 就是“真正产生智能”的地方。我们可以把主脑的思考过程拆解为 4 个清晰的动作：

### 1. 最后的确认：触发“缝合手术”

在 `forward` 函数的一开头，你就会看到这行熟悉的代码：

Python

```
input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(...)
```

没错，这里就是 **Chapter 6 的真正发源地**！主干网络在正式思考前，先调用了这个函数，把刚才我们聊过的“移花接木”手术做完，拿到了纯正的特征向量 `inputs_embeds`。

### 2. 深入思考：进入 GPTNeoX 大脑

接下来，整条链路最消耗算力、最核心的代码出现了：

Python

```
outputs = self.get_model()(
    input_ids=input_ids,
    attention_mask=attention_mask,
    inputs_embeds=inputs_embeds,
    # ...
)
```

- **发生了什么**：这里的 `self.get_model()` 其实就是 `GPTNeoXModel`（一个庞大的语言模型）。这串融合了图文的向量被送进了几十层的 Transformer 神经网络中。
    
- **Attention 机制发威**：在这里，模型开始疯狂地计算注意力。指令里的“ball”（球）这个词的向量，会去和几百个图像特征向量产生互动，模型通过这种复杂的矩阵相乘，最终“领悟”到：**“哦，主人说的球，就是画面左下角的那个红色圆形像素块！”**
    

### 3. 提取智慧结晶：`hidden_states`

思考完毕后，大脑吐出了一个极其关键的变量：

Python

```
hidden_states = outputs[0]
```

- **它的意义**：这是语言模型对当前“所见+所闻”的**最高维度的理解**。它不再是简单的像素或单词，而是一份包含了“空间关系、物体识别、任务意图”的**综合战略报告**。后续的动作生成，全都要指望这份报告。这个hidden_state就是论文里面说的条件embeding，也就是**就是动作头（Action Head）在生成动作时所依赖的“条件嵌入（Conditional Embedding）”**。
    

### 4. 任务分发：呼叫下游“动作部门”

大脑（GPTNeoX）只负责“理解”，不负责“动手”。所以 `forward` 函数的最后部分，变成了一个**路由器**：

Python

```
elif self.head_type == 'act':
    if not eval: ...
    else:
        action = self.forward_act_head(actions, hidden_states, states, is_pad)
        return action
elif self.head_type == 'droid_diffusion':
    # ... 调用 forward_diffusion_head
```

- 代码会根据你配置的 `head_type`，把刚才算出来的 `hidden_states`（战略报告）以及 `states`（机器人当前的真实关节状态），下发给具体的动作头（比如 ACT 或 Diffusion）。
    
- 因为我们现在是推理模式（`eval=True`），所以代码直接返回了预测出的 `action`，而不再去计算什么 loss（误差）了。

# Chapter 8. Diffusion 动作头：从噪声里一步步雕出动作
在 **Chapter 7** 中，主脑（CEO）已经把那份极其关键的“战略报告”——也就是你提到的条件 Embedding（`hidden_states`）交到了你手上。
现在，**Chapter 9** 的 **Diffusion 动作头** 将接手这份报告，开启一场名为“从混沌中雕刻动作”的艺术表演。我们可以对照 `llava_pythia_3.py` 中 `forward_diffusion_head` 的推理部分（Line 296-312）来拆解这个过程：
### 1. 制造“混沌”：初始化纯噪声数据 (Line 301)

Python

```
noisy_action = torch.randn((B, Tp, action_dim)).cuda()
```

- **动作**：Diffusion 的推理不是直接猜动作，而是先随机生成一团毫无规律的“高斯噪声”。
    
- **物理意义**：这团噪声代表了无限种可能的动作序列（形状为 `[Batch, 预测步数, 动作维度]`）。模型接下来的任务，就是把这团乱麻一点点理顺。
    

### 2. 设定“雕刻步数” (Line 305)

Python

```
self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
```

- **动作**：调用 `DDIMScheduler`（调度器），将推理过程划分为若干步（代码中是 $10$ 步）。
    
- **意义**：这决定了机器人“思考”的细腻程度。步数越多，动作往往越平滑，但计算耗时也会增加。
    

### 3. 核心大戏：迭代去噪循环 (Line 307-312)

这是 Diffusion 最神来之笔的地方。程序会进入一个 `for` 循环，利用主脑给的 `hidden_states`（条件）来去噪：

Python

```
for k in self.noise_scheduler.timesteps:
    # A. 预测噪声
    noise_pred = self.embed_out(naction, k, global_cond=hidden_states, states=states)
    
    # B. 剔除噪声
    naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
```

- **A. 预测噪声 (`self.embed_out`)**：这里调用了 `ConditionalUnet1D`。（在`LlavaPythiaForCausalLM` 类的 **`__init__` 初始化函数**里面有写到，在代码的第 58 到 62 行，
- self.embed_out = ConditionalUnet1D( input_dim=config.action_dim, global_cond_dim=config.hidden_size, state_dim=config.state_dim )  , 所以当你执行 `noise_pred = self.embed_out(naction, k, global_cond=hidden_states, states=states)` 时，实际上是在调用 `ConditionalUnet1D` 的 `forward` 方法。
    
    - 它盯着这团乱糟糟的动作 `naction`，同时参考主脑给出的**条件 Embedding (`hidden_states`)** 和机器人现在的**姿态 (`states`)**。
        
    - 它的脑回路是：“基于现在的图像（条件），这个乱七八糟的动作里，哪部分是‘错误’的噪声？”
        
- **B. 剔除噪声 (`self.noise_scheduler.step`)**：调度器根据预测结果，把那部分“错误”减掉，得到一个比刚才稍微“清爽”一点点的动作序列 `prev_sample`。
    

### 4. 终点：破茧而出的动作序列

经过 $10$ 轮循环的“洗礼”，原本的随机噪声已经彻底变成了一段极其合理的**动作轨迹 (`naction`)**。

---

### 💡 为什么 Diffusion 头需要 Chapter 7 的那个 Embedding？

如果没有 Chapter 7 提供的 `hidden_states`，这个去噪过程就失去了指南针。

- **没有条件**：模型只会生成一段“像人动的轨迹”，但不知道是该去抓球还是去开门。
    
- **有了条件**：在每一轮去噪时，`ConditionalUnet1D` 都会被提醒：“现在的任务是抓那个球，图片里球在左边！” 这样，它剔除噪声的方向就会引导动作向“向左抓取”的方向靠拢。

# chapter9.时间聚合：让大家一起投票
真实机器人执行不能只信一次预测，因为模型每一帧都会重新看世界，前后几次动作也会有一点差异。时间聚合的目的，就是把最近几轮预测做平滑融合，让动作更稳，不至于抖来抖去。
这一章（Chapter 10）它的学术名词叫 **“时间聚合 (Temporal Aggregation)”** 或者叫 **“动作融合 (Action Ensembling)”**。

如果不理解它的背景，看这些矩阵切片操作就像看天书。我们先抛开代码，用最通俗的比喻来理解它的**物理意义**。

### 1. 为什么要“时间聚合”？

一般的机器人模型是“走一步看一步”：看一眼图，预测当前这一步的动作。 但在你使用的这种先进架构（ACT 或 Diffusion）中，模型采用了 **Action Chunking（动作切块）** 技术。这意味着，它每看一眼，**不仅预测当前的动作，还一口气预测了未来几十步的动作轨迹**（也就是 `num_queries`）。

这就会产生一个奇妙的现象——**“历史的重叠”**：

- **第 0 秒**：模型预测了 0~49 秒的动作。
    
- **第 1 秒**：模型预测了 1~50 秒的动作。
    
- **第 2 秒**：模型预测了 2~51 秒的动作。
    

当真实时间来到**第 2 秒**时，机器人到底该执行哪个动作？

它手里现在有三份关于第 2 秒的“预测报告”：第 0 秒猜的、第 1 秒猜的、以及刚刚第 2 秒最新猜的。

**时间聚合的目的，就是不盲目只听最新的一次，而是把过去所有针对“当前时刻”的预测拿出来，大家一起“加权投票”，得出一个最平滑、最稳的动作**。这样可以极大减少机械臂在执行时的“手抖”现象。

---

### 2. 代码的核心动作拆解（The How）

结合你发的截图和代码（第 334 行开始），这场“投票大会”分四步进行：

#### 第一步：建一个巨大的“历史记录本”

Python

```
all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, action_dim], ...)
```

代码在 GPU 里开辟了一个巨大的三维空表。 每当模型输出一段未来的动作块（`all_actions`），代码就会把它填进 `all_time_actions[[t], t:t + num_queries]` 这个位置。这就好比把每一份“预测报告”都按时间顺序归档。

#### 第二步：把关于“现在”的所有报告都抽出来

Python

```
actions_for_curr_step = all_time_actions[:, t]
```

当时间来到 `t` 时，代码拿着刀在这个三维表上**竖着切了一刀**。 它把过去所有时间点（`[:, t]`）对当前时刻 `t` 做出的动作预测全都拿了出来。`actions_populated` 则是为了剔除掉那些还没填入数据的空白行。

#### 第三步：论资排辈（指数衰减权重）

Python

```
k = 0.01
exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
```

既然大家都要投票，那谁的票权更重？ 直觉告诉我们：**越新的预测越准（因为它看到了最新的画面），越老的预测越不准**。 所以代码引入了指数衰减公式 $e^{-k \cdot x}$。

- 最新预测（步数差为 0）：权重最高。
    
- 越早的预测（步数差越来越大）：权重呈指数级下降。
    
- `k = 0.01` 就是用来控制这个“遗忘速度”的。
    

#### 第四步：加权汇总，得出最终动作

Python

```
raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
```

把每个历史动作乘以它的权重，然后全部相加（加权平均）。 这个算出来的 `raw_action`，就是融合了过去所有智慧的**最平滑的最终动作**！

# Chapter 10. 动作后处理：把模型答案变成人类世界能执行的指令

模型输出的动作，还只是“神经网络语言”。机械臂真正能执行的，是带单位、带姿态约定、带范围约束的控制量。所以最后必须把动作反归一化，再把旋转表示转成机器人控制器能读懂的格式。

### ⚙️ 核心动作 (How)

1. `stats = dataset_stats.pkl` 里保存了动作的均值、标准差、最值。
2. 如果是 `act` 头，执行 `a * std + mean`。
3. 如果是 `transformer_diffusion`，把 `[-1, 1]` 映射回 `[min, max]`。
4. `convert_actions(pred_action)` 里把动作拆成三段：
    - `xyz`
    - `rot6d`
    - `gripper`
5. `rot_6d_to_euler_angles(...)` 把 6D 旋转转成欧拉角。
6. 最终拼成 `xyz + euler + gripper`，变成可下发的动作向量。

# Chapter 11. 真实下发：机械臂做出最后一刻动作


这是整条链路的终点。前面所有计算，最后都要收束成一次真实的环境步进。模型不再“想”，而是让机械臂真的动起来。到这一刻，虚拟推理变成了物理世界的反馈。

### ⚙️ 核心动作 (How)

1. `raw_action` 经过后处理后变成最终 `action`。
2. `deploy_env.step(action)` 把动作发给真实机械臂环境。
3. 环境返回 `action_info`，完成一次控制闭环。
4. 循环继续，下一帧再采样、再推理、再下发，直到任务结束。


# 总结：
### ⚙️ 核心动作 (How)

1. `eval_real_franka.py` 启动，加载 policy。
2. `load_pretrained_model(...)` 把 VLM、视觉塔、投影层、动作头装好。
3. `deploy_env.get_observation()` 取回图像和机器人状态。
4. 图像归一化、补方、拆左右视角；状态转张量。
5. `raw_lang` 和视觉 token 拼成 prompt。
6. `prepare_inputs_labels_for_multimodal(...)` 把图像特征插进文本序列。
7. `GPTNeoXModel` 产出 hidden states。
8. `act` 或 `diffusion` 动作头把 hidden states 变成动作 chunk。
9. 时间聚合把多个 chunk 做平滑融合。
10. 动作反归一化、旋转表示转换。
11. `deploy_env.step(action)` 下发到真实机械臂。
12. 机器人执行动作，完成闭环。
