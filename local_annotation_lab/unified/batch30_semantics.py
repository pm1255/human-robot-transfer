"""Visual pilot labels. Boundaries are uncertain; source instructions are retained separately."""
from pathlib import Path
import json
R=Path('/Users/panmiao/Documents/ChatGPT/paper/outputs/unified_30_20261021')
# End time, shared phase, shared verb, observed segment description.
CONFIG={
'abc130k_ep_010392':('取下短裤并放入篮子','cloth','container',1.2,[(1,'approach','reach','夹爪接近悬挂衣物'),(4.6,'manipulate','unhang','抓持衣架并将衣物从横杆取下'),(10.5,'manipulate','separate','双臂调整衣物，分离衣架与衣物'),(14.3,'transport','move','左臂将衣物移向篮子，右臂持衣架'),(16.1,'release','place','将衣物放入篮子'),(None,'retract','withdraw','双臂从篮子上方撤离')]),
'abc130k_ep_014083':('依次将塑料瓶放入桶内','bottle','bin',1.1,[(1.4,'approach','reach','左夹爪接近桌面瓶子'),(3.6,'transport','move','左夹爪将首个瓶子提起放入桶'),(4.5,'approach','reach','左夹爪转向绿色瓶子'),(6.3,'transport','place','将绿色瓶子移入桶中'),(8.1,'transport','place','右夹爪将红色瓶子移入桶中'),(10.6,'transport','place','左夹爪将蓝色瓶子移入桶中'),(12.4,'transport','place','右夹爪将透明瓶子移入桶中'),(14.5,'approach','reach','左夹爪接近最后一个透明瓶子'),(None,'transport','place','左夹爪提起最后一个瓶子移向桶内')]),
'abc130k_ep_030633':('从塑料袋中逐次取出仿真水果','plastic_bag','tray',1.5,[(1.2,'approach','reach','双夹爪接近塑料袋口'),(4.4,'manipulate','open','撑开并调整塑料袋开口'),(7.1,'transport','place','左夹爪从袋中取出水果移向左侧托盘'),(9.3,'approach','reach','左夹爪再次伸向袋内'),(11.8,'transport','place','再次取出水果放向托盘'),(14.2,'approach','reach','调整袋口并伸向袋内'),(16.8,'transport','place','取出另一件水果移向托盘'),(18.4,'approach','reach','左夹爪再次接近袋口'),(21.7,'transport','place','取出剩余可见水果并放到托盘'),(None,'retract','withdraw','双夹爪松开袋子并撤离')]),
'abc130k_ep_044053':('整理耳机线缆','headphones','headphones',1.3,[(1.8,'approach','reach','双夹爪接近耳机与线缆'),(3.8,'transport','lift','提起耳机并拉开线缆'),(10.7,'manipulate','wrap','反复调整耳机与线缆相对位置'),(12.1,'release','place','将耳机暂放桌面'),(17.8,'manipulate','wrap','重新抓持耳机并继续整理线缆'),(19.1,'release','place','将耳机放回桌面'),(None,'retract','withdraw','双夹爪松开撤离')]),
'abc130k_ep_044783':('整理调味瓶至托盘','bottle','tray',1.2,[(2.8,'align','adjust','调整双夹爪与调味瓶的相对位置'),(5.7,'transport','place','左夹爪将一只调味瓶移到红色托盘'),(9.5,'transport','place','右夹爪抓起右侧瓶子并移向托盘'),(13.6,'manipulate','arrange','双夹爪整理托盘中的瓶子'),(16.5,'transport','place','将剩余调味瓶移到托盘旁'),(17.8,'align','arrange','调整瓶子的排列'),(None,'retract','withdraw','双夹爪离开整理区域')]),
'abc130k_ep_082354':('将餐盘装入沥水架','plate','rack',1.1,[(1.1,'approach','reach','右夹爪接近右侧餐盘'),(2.7,'transport','lift','提起首个餐盘'),(4.6,'align','insert','竖起餐盘并放入沥水架'),(6.9,'manipulate','grasp','双夹爪协作拿起第二个餐盘'),(10.8,'align','insert','调整第二个餐盘并放入架中'),(12.8,'approach','reach','夹爪转向桌前的第三个餐盘'),(14.7,'transport','lift','提起第三个餐盘'),(None,'align','insert','将第三个餐盘竖起对准沥水架')]),
'abc130k_ep_096887':('将仿真水果放入果盘','fruit','bowl',.65,[(1,'approach','reach','双夹爪接近桌面水果'),(2.4,'transport','place','抓起两件水果移向果盘'),(4.6,'release','place','放下水果并调整其位置'),(5.5,'approach','reach','夹爪接近剩余的绿色水果'),(7.6,'transport','place','将绿色水果放进果盘'),(8.3,'align','arrange','整理果盘内的水果'),(None,'retract','withdraw','双夹爪撤离')]),
'umi_hifi_microwave':('从微波炉取出餐盒','container','microwave',1.2,[(2.8,'idle','wait','夹爪位于身前等待'),(4.7,'approach','reach','右夹爪接近微波炉门'),(5.8,'manipulate','open','拉开微波炉门'),(8.6,'manipulate','grasp','右夹爪伸入微波炉并夹住餐盒'),(11.8,'transport','move','将餐盒移出微波炉'),(14.5,'align','adjust','把餐盒移到身前调整'),(None,'idle','hold','在身前保持餐盒')]),
'umi_hifi_tray':('用抹布擦拭托盘','cloth','tray',1.1,[(2.6,'transport','move','右夹爪持抹布移到托盘上方'),(3.4,'contact','touch','将抹布压向托盘'),(14.8,'manipulate','wipe','持抹布反复擦拭托盘表面'),(16,'release','place','松开抹布并留在托盘内'),(None,'retract','withdraw','右夹爪从托盘撤离')]),
'umi_hifi_pillow':('将枕巾铺到枕头上','cloth','pillow',1.2,[(1.4,'approach','reach','双夹爪接近枕巾两端'),(3.1,'manipulate','grasp','先后夹住枕巾两端'),(7,'manipulate','unfold','提起并横向展开枕巾'),(10.8,'align','align','将展开的枕巾对准枕头'),(13.2,'transport','place','把枕巾放到枕头表面'),(14.7,'release','release','松开两端枕巾'),(None,'retract','withdraw','双夹爪回到身前')]),
'umi_hifi_folder':('将文件夹翻转立起','folder','table',.95,[(2.4,'idle','wait','夹爪在文件夹前等待'),(4.3,'approach','reach','双夹爪接近文件夹两侧'),(5.6,'manipulate','grasp','夹住并提起文件夹边缘'),(8.1,'manipulate','rotate','翻转文件夹使其立起'),(10.1,'release','place','把文件夹稳定放在桌面后松开'),(None,'retract','withdraw','双夹爪退回身前')]),
'umi_fast_suitcase':('收纳瓶子并合上箱包','suitcase','suitcase',1.2,[(1.7,'idle','wait','夹爪在闭合箱包前等待'),(4.2,'manipulate','open','接近箱包并掀开上盖'),(5.8,'approach','reach','右夹爪转向桌边瓶子'),(9.8,'transport','place','夹住瓶子并放入敞开的箱包'),(11.4,'release','release','松开瓶子并离开箱包内部'),(14.3,'manipulate','close','压下箱包上盖'),(None,'retract','withdraw','夹爪从合起的箱包撤离')]),
'umi_fast_rice':('打开电饭煲并向内添加物料','rice_cooker','rice_cooker',1.4,[(2,'idle','wait','夹爪在电饭煲前等待'),(4,'approach','reach','左夹爪接近电饭煲开盖按钮'),(5.1,'manipulate','open','按动按钮并打开锅盖'),(8.1,'manipulate','grasp','右夹爪从右侧取起勺具'),(10.1,'transport','move','将勺具移到电饭煲上方'),(12,'manipulate','pour','向锅内倾倒勺中物料'),(14.5,'release','place','将勺具移回右侧并放下'),(17.6,'manipulate','close','压下电饭煲锅盖'),(None,'retract','withdraw','夹爪退回身前')]),
'umi_fast_sandwich_003103':('将生菜和面包叠成三明治','lettuce','bread',1.1,[(1.6,'idle','wait','夹爪在食材前等待'),(2.8,'approach','reach','夹爪接近生菜'),(5.8,'transport','place','夹起生菜并移到面包所在的蓝盘'),(7,'release','release','松开生菜，使其留在盘上'),(8.8,'approach','reach','夹爪转向左侧面包盒'),(10.8,'manipulate','grasp','夹起面包片'),(13.4,'transport','place','将面包片移向生菜并放下'),(15.6,'release','release','调整并松开面包片'),(None,'retract','withdraw','夹爪离开盘子')]),
}
# Ego source subtasks remain in source.json. These reviewed drafts can refine or disagree.
CONFIG.update({
'egosuite_p04_ep_028028':('将货架上的小瓶摆正','bottle','shelf',.9,[(1.4,'manipulate','adjust','双手调整前排倒下的小瓶'),(3.6,'manipulate','upright','右手扶正右侧小瓶'),(6.9,'manipulate','upright','右手依次扶正中间小瓶'),(9.8,'manipulate','upright','扶正左侧小瓶并调整间距'),(13.8,'align','arrange','双手整理一排小瓶使其朝前'),(None,'retract','withdraw','双手开始从货架撤离')]),
'egosuite_p04_ep_021026':('收纳遥控器与钥匙扣','container','container',.7,[(1.4,'align','place','将透明收纳盒摆到桌面中间'),(2.9,'transport','place','右手将遥控器放入盒中'),(5.3,'transport','place','把红色钥匙扣放入盒中'),(7.1,'manipulate','close','拿起透明盖并盖到盒上'),(9,'manipulate','press','双手按压盒盖使其扣合'),(10.4,'transport','move','将收纳盒移到桌角'),(None,'unknown','unknown','源描述未覆盖的末帧')]),
'egosuite_p04_ep_036617':('将盒装商品补到货架','package','shelf',.75,[(1.4,'manipulate','grasp','双手从购物车拿起盒装商品'),(3.4,'transport','place','把商品摆到货架上'),(5,'manipulate','grasp','再次从购物车拿起盒装商品'),(7,'transport','place','将第二批商品放到货架'),(11.2,'align','arrange','双手整理货架前排商品'),(None,'unknown','unknown','源描述未覆盖的末帧')]),
'egosuite_p05_ep_033793':('取抹布清洁椅旁区域','cloth','chair',.85,[(1.4,'manipulate','grasp','右手从托盘取起抹布'),(5.6,'manipulate','wipe','右手向椅下区域擦拭，接触部位部分遮挡'),(10.5,'transport','move','抹布移回托盘附近并调整持握'),(12.4,'release','place','将抹布放回托盘'),(None,'unknown','unknown','源描述未覆盖的末帧')]),
'egosuite_p05_ep_023134':('从冰箱取水并关门','bottle','countertop',.9,[(1.6,'manipulate','open','拉开冰箱门'),(2.4,'approach','reach','左手伸向冰箱内水瓶'),(3.6,'manipulate','grasp','抓住并取出水瓶'),(6.3,'transport','move','持水瓶转向灶台旁'),(7.6,'release','place','将水瓶放到台面后松手'),(9.8,'retract','turn','转身返回冰箱前'),(13.1,'manipulate','close','双手推动冰箱门关闭'),(None,'retract','withdraw','手离开关闭的冰箱门')]),
'egosuite_p05_ep_029284':('用抹布擦拭壁画边框','cloth','picture_frame',.75,[(1.4,'manipulate','grasp','从椅上拿起黄色抹布'),(3.1,'transport','move','持抹布靠近壁画边框'),(8.2,'manipulate','wipe','反复擦拭壁画边框'),(10.0,'transport','move','移开抹布并返回椅子前'),(None,'release','place','把抹布放回椅面并松开')]),
'egosuite_p06_ep_004360':('整理纸杯与杯垫','cup','table',1.1,[(5.7,'manipulate','stack','双手依次叠起桌上的纸杯'),(6.8,'release','place','将第一叠纸杯放到右侧'),(11.7,'manipulate','stack','继续收集并叠放剩余纸杯'),(12.9,'release','place','把第二叠纸杯放到右侧'),(14.4,'manipulate','stack','整理桌面剩余杯子'),(22.0,'manipulate','stack','双手收集并叠放杯垫'),(None,'retract','withdraw','杯垫靠拢后双手准备撤离')]),
'egosuite_p06_ep_022668':('擦拭衣柜挂杆后关门','cloth','cabinet',1.2,[(1.5,'manipulate','grasp','从盆内取起抹布'),(3,'approach','reach','持抹布移到衣柜挂杆'),(8.4,'manipulate','wipe','沿横杆反复擦拭'),(10.6,'transport','place','把抹布放回盆内'),(12.7,'approach','reach','双手移向衣柜门'),(19,'manipulate','close','依次推动柜门关闭'),(None,'unknown','unknown','源描述未覆盖的末帧')]),
'egosuite_p07_ep_021574':('依次拉开两侧窗帘','curtain','window',.9,[(.7,'contact','grasp','双手抓住左侧窗帘'),(5.9,'manipulate','open','将左侧窗帘拉向侧边'),(6.7,'approach','reach','双手换到右侧窗帘'),(12.8,'manipulate','open','将右侧窗帘推向侧边'),(13.2,'retract','withdraw','双手离开窗帘'),(None,'unknown','unknown','源描述未覆盖的末帧')]),
'egosuite_p07_ep_033567':('依次用三支记号笔画线','marker','paper',1.3,[(1.7,'manipulate','uncap','拿起蓝色记号笔并拔开笔帽'),(4.6,'manipulate','draw','用蓝色笔在纸上画线'),(6.8,'manipulate','cap','盖上蓝色笔并放回笔筒'),(8.5,'manipulate','uncap','拿起红色笔并拔开笔帽'),(10.6,'manipulate','draw','用红色笔在纸上画线'),(12.9,'manipulate','cap','盖上红色笔并放回笔筒'),(14.4,'manipulate','uncap','拿起黑色笔并拔开笔帽'),(17,'manipulate','draw','用黑色笔在纸上画线'),(19.6,'manipulate','cap','盖上黑色笔并移回笔筒'),(None,'unknown','unknown','源描述未覆盖的末帧')]),
})
def refine_objects(sid, segments):
 for s in segments:
  label=s['label']
  if sid=='egosuite_p05_ep_023134' and s['verb']in ['open','close','withdraw']:
   s.update(object='refrigerator_door',target='refrigerator')
  if sid=='egosuite_p05_ep_029284' and s['verb']=='place':s['target']='chair'
  if sid=='egosuite_p06_ep_004360' and '杯垫'in label:s['object']='coaster'
  if sid=='umi_hifi_microwave' and s['verb']=='open':s.update(object='microwave_door',target='microwave')
 return segments

if __name__=='__main__':
 (R/'semantic_config.json').write_text(json.dumps(CONFIG,ensure_ascii=False,indent=2))
