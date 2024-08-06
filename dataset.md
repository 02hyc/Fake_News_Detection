数据集

crowdsourcing

数据集 自然背景下收集

人工标注，打上标签 true half true false

需要包括辅助信息，例如社交媒体上的用户社交活动，以帮助做出决定。

虚假新闻可能会在不正确的背景下引用真实证据来支持非事实性主张[22]。因此，现有的手工制作和数据特定的文本特征通常不足以用于假新闻检测。还必须应用其他辅助信息来改进检测，例如知识库和用户社交活动。

“假新闻”一词的定义达成一致，进行适当分类

###### 假新闻定义：

狭义定义是新闻文章，这些新闻文章故意和严重错误，可能会误导读者[2]。这个定义有两个关键特征：真实性和意图。以不诚实的意图误导消费者而产生的。将欺骗性新闻视为假新闻[66]，其中包括严肃的捏造，恶作剧和讽刺。

以下概念不是假新闻：（1）具有适当背景的讽刺新闻，其无意误导或欺骗消费者，并且不太可能被误认为是事实; （2）不是来自新闻事件的谣言; （3）阴谋论，不论是真还是假; （4）无意中产生的错误信息; （5）仅以娱乐为动机或欺骗目标个人的恶作剧。

虚假新闻的媒体生态一直在变化

发布者和消费者。新闻发布的过程被建模为从原始信号s到结果新闻报道a的映射，其具有失真偏差b的效果，即![img](https://tongtianta.site/oss/paper_image/e05e1d3a-f188-11e8-a816-00163e08bb86/INNERLATEXT220.jpg)，其中b = [-1,0,1]表示[lef t，no，right ]偏见对新闻发布过程产生影响。直观地说，这正在捕捉新闻文章可能被偏见或扭曲以产生假新闻的程度

消费者有选择地接触某些类型的新闻，因为新闻提要在社交媒体的主页上出现，放大了消除上述假新闻的心理挑战。

###### 假新闻检测

新闻文章。它由两个主要组件组成：Publisher和Content 社交新闻参与 以表示新闻在n中随时间传播的过程用户U = ![img](https://tongtianta.site/oss/paper_image/e05e1d3a-f188-11e8-a816-00163e08bb86/INNERLATEXT323.jpg) 1 ![img](https://tongtianta.site/oss/paper_image/e05e1d3a-f188-11e8-a816-00163e08bb86/INNERLATEXT324.jpg) 2 ![img](https://tongtianta.site/oss/paper_image/e05e1d3a-f188-11e8-a816-00163e08bb86/INNERLATEXT325.jpg)及其相应的帖子P = ![img](https://tongtianta.site/oss/paper_image/e05e1d3a-f188-11e8-a816-00163e08bb86/INNERLATEXT326.jpg) 1 ![img](https://tongtianta.site/oss/paper_image/e05e1d3a-f188-11e8-a816-00163e08bb86/INNERLATEXT327.jpg) 2 ![img](https://tongtianta.site/oss/paper_image/e05e1d3a-f188-11e8-a816-00163e08bb86/INNERLATEXT328.jpg)在社交媒体上关于新闻文章a

将假新闻检测定义为二元分类问题：假新闻本质上是对发布者操纵的信息的失真偏差。

一种用于假新闻检测的通用数据挖掘框架，其包括两个阶段：（i）特征提取和（ii）模型构建。特征提取阶段旨在以正式的数学结构表示新闻内容和相关辅助信息，并且模型构建阶段进一步构建机器学习模型以更好地基于特征表示来区分假新闻和真实新闻。

从新闻内容和社交背景中提取和表示有用功能的细节。

语言特征 视觉特征 社会背景特征 通常，我们想要表示的社交媒体上下文有三个主要方面：用户，生成的帖子和网络。

提取各个级别的特征以使用用户人口统计的各个方面来推断每个用户的可信度和可靠性，例如注册年龄，关注者/被跟随者的数量，用户创作的推文的数量等[11]。

提取基于帖子的特征以通过帖子中表达的公众反应来帮助发现潜在的假新闻是合理的。基于帖子的功能侧重于识别有用信息，以从相关社交媒体帖子的各个方面推断新闻的准确性。

用户在兴趣，主题和关系方面在社交媒体上形成不同的网络。如前所述，假新闻传播过程倾向于形成回声室周期，突出了提取基于网络的功能以表示用于虚假新闻检测的这些类型的网络模式的价值

模型构建

基于知识的方法旨在使用外部资源来检查新闻内容中的建议声明。事实检查的目标是在特定情境中为声明分配真值[83]。

基于立场：基于立场的方法利用来自相关帖子内容的用户观点来推断原始新闻文章的准确性。使用这些方法，我们可以根据相关帖子的立场价值来推断新闻的准确性。

基于传播的：基于传播的假新闻检测方法，用于关联相关社交媒体帖子的相互关系以预测新闻可信度。

###### 评估检测效率

评估指标，将其视为一个分类问题

• True Positive (TP): when predicted fake news pieces are actually annotated as fake news;

•真正的正面（TP）：当预测的假新闻片实际上被注释为假新闻时;

• True Negative (TN): when predicted true news pieces are actually annotated as true news;

•真正的否定（TN）：当预测的真实新闻片段实际上被注释为真实新闻时;

• False Negative (FN): when predicted true news pieces are actually annotated as fake news;

•假否定（FN）：当预测的真实新闻片实际上被注释为假新闻时;

• False Positive (FP): when predicted fake news pieces are actually annotated as true news.

•误报（FP）：当预测的假新闻片实际上被注释为真实新闻时。

![image-20230711185051251](C:\Users\13414\AppData\Roaming\Typora\typora-user-images\image-20230711185051251.png)

F1用于结合精确度和召回率，可以为假新闻检测提供整体预测性能。请注意，对于P recision，Recall，F1和Accuracy，值越高，性能越好。



相关工作

![image-20230711190419561](C:\Users\13414\AppData\Roaming\Typora\typora-user-images\image-20230711190419561.png)

![image-20230711191950412](C:\Users\13414\AppData\Roaming\Typora\typora-user-images\image-20230711191950412.png)

![image-20230711192706810](C:\Users\13414\AppData\Roaming\Typora\typora-user-images\image-20230711192706810.png)

![image-20230711193127601](C:\Users\13414\AppData\Roaming\Typora\typora-user-images\image-20230711193127601.png)