### Deep research agent

**Goal: DR Agent完成带引用的long-form reports和short-from answers. 强调DR Agent通过连续地使用搜索工具并不断地调整其搜索广度和搜索深度获得精确的问题回答和深度的长篇报告**

![dr_agent.png](./dr_agent.png)

### Agent architecture

Agent loop采用主副架构, 不使用multi-agent架构, 减少LLM引入的不可控. 

1. agent的preloop, human in the loop: 设定搜索目标, 计划, 与用户对齐
2. summary agent as tool: 总结web, paper, locals返回的内容, 填充对应的模版, 作为main agent的唯一输入; 目的是为了减少上下文消耗, 去掉无关信息
3. main agent: 根据0设定的目标和计划, 判断1的模版输入值得保留的部分, 以及需要探索的部分. 保留可用的部分(offloading), 确定下一次的搜索循环.

### Todos

基于本地部署的Qwen3-8B的DR Agent

- [x]  开源推理模型本地部署: vllm, qwen3-8B/qwen2.5-7B-instruct
    
    CUDA_VISIBLE_DEVICES=1 vllm serve /home/ubuntu/workspace/dr_agent/qwen3-8B \
    
    --served-model-name qwen3-8b \
    
    --enable-auto-tool-choice \
    
    --tool-call-parser hermes
    
    CUDA_VISIBLE_DEVICES=0,1 vllm serve /home/ubuntu/workspace/dr_agent/qwen3-8B --served-model-name qwen3-8b --enable-auto-tool-choice --tool-call-parser hermes --tensor-parallel-size 2
    
- [ ]  搭建DR Agent: openai agent sdk + vllm
- [ ]  记录DR Agent的tools tracing和reasoning tracing: openai agent sdk
- [ ]  构建推理数据集 + Multi-Run 工具调用数据集: wandb
- [ ]  Prompt Evaluation: 评估数据集
- [ ]  使用GPT5, Gemini等推理模型来模仿DR Agent来生成推理数据集
- [ ]  先做SFT来提升模型的基本能力: llama
- [ ]  使用部署的推理模型做Reasoning RL的验证实验: verl
- [ ]  最后在推理模型上做Agentic RL: rllm

**Features:** 

1. Web search
2. Papaer search
3. Local doc lookup
4. Human in the loop
5. Planning

**一些分散的指标:**

- 搜索数量上, 1)agent的查询的网站的总量, 2)agent的正文的引用量
- 搜索质量上, agent查询的网站的好坏
    - 避免内容农场和一些公认的垃圾网站
    - 尽可能避免使用任何来自于中文互联网的内容
- 报告的客观质量指标: 报告的全文长度, 详细程度
- 报告引用的可靠性: 1) 是否与查询的网站对应, 2)是否正确引用了查询网站的内容, 而不是自己编造
- DR benchmark: 参考DR Tulu的几个benchmark

### DR_Agent tools designation

> tools的设计逻辑 +  对应的tools prompts
> 
1. 网页检索+浏览, 目的是找到相关的信息, 广度高于深度
    1. 最主要的搜索API用 duckduckgo search, 保证稳定性
    2. 只用前5/10/20个搜索结果, 允许agent用不同的关键词多次检索, 除非用户指定, 否则一般都不直接用用户提供的搜索关键词来做检索. 搜索工具的的system prompt除了详细说明工具如何使用之外的, 可能还需要去具体说明一些关键词的检索技巧. 
    3. **将网页的内容从复杂的html, css中提取出来转换为markdown的格式**
    4. 最大程度保持工具的原子性和工具的灵活性, 不内置复杂和特殊的逻辑, 减少需要的前置条件. 目的是为了让agent重复利用这些基础的工具来完成任务, 最大程度地依赖agent自身能力的发挥. 具体实现逻辑:
        1. 关键词检索, 搜索引擎返回前n个相关的URL, 由Agent决定广度
        2. 抓取URL的全部内容, 用正文模版做内容提取和格式(多余空格和空行)后处理
        3. Agent as tool: 将全部的正文content委派给n个agent总结并按照固定格式(summary schema)返回全部内容给主Agent, 每个网页的最大使用长度由Agent决定, 即一次tool calling
2. 本地文档查询: 
    1. 如果存在本地文档, 那么优先使用本地文档
    2. 在Human in the loop 阶段提前使用本地文档做好用户意图的识别和deep research的整个流程的planning.
    3. 上传的全部文档转换为markdown格式
3. 论文查询和检索(两种检索方式: precise mode 和 broad mode)
    1. 粗检索: 使用ddgs(duckduckgo, google search)来模糊检索论文的metadata(url, title, doi等)
        1. 如果论文来源包含在诸如arxiv等openacess网站list中, 则直接获取全部信息(包括pdf直链)
    2. 论文的主要获取点: semantic scholar, openalex, unpaywall获得论文的全部信息
    3. Agent as tool: 将论文pdf 转换为markdown形式委派给n个agent总结并按照固定格式(summary schema)返回全部内容给主Agent

### Human in the loop

1. 意图识别
    1. 使用local files loopkup补充问题上下文
    2. 对其需求,用户确认.
    3. handoff到搜索规划
2. 搜索规划
    1. 针对每个问题制定具体搜索目标
    2. 每个目标对应一个计划点todo
    3. context offloading: 整个todo计划表download到context之外
        1. 由agent决定目标进度
        2. 每一次的工具调用之后确认计划表

---

###