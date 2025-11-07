import json
import re
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api import AstrBotConfig


@dataclass
class EmotionalState:
    """情感状态数据类"""
    # 基础情感维度
    joy: int = 0
    trust: int = 0
    fear: int = 0
    surprise: int = 0
    sadness: int = 0
    disgust: int = 0
    anger: int = 0
    anticipation: int = 0
    
    # 复合状态
    favor: int = 0  # 综合好感度
    intimacy: int = 0  # 亲密度
    
    # 关系状态
    relationship: str = "陌生人"
    attitude: str = "中立"
    
    # 行为统计
    interaction_count: int = 0
    last_interaction: float = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    
    # 用户设置
    show_status: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotions": {
                "joy": self.joy,
                "trust": self.trust,
                "fear": self.fear,
                "surprise": self.surprise,
                "sadness": self.sadness,
                "disgust": self.disgust,
                "anger": self.anger,
                "anticipation": self.anticipation
            },
            "states": {
                "favor": self.favor,
                "intimacy": self.intimacy
            },
            "relationship": self.relationship,
            "attitude": self.attitude,
            "behavior": {
                "interaction_count": self.interaction_count,
                "last_interaction": self.last_interaction,
                "positive_interactions": self.positive_interactions,
                "negative_interactions": self.negative_interactions
            },
            "settings": {
                "show_status": self.show_status
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalState':
        state = cls()
        if "emotions" in data:
            emotions = data["emotions"]
            state.joy = emotions.get("joy", 0)
            state.trust = emotions.get("trust", 0)
            state.fear = emotions.get("fear", 0)
            state.surprise = emotions.get("surprise", 0)
            state.sadness = emotions.get("sadness", 0)
            state.disgust = emotions.get("disgust", 0)
            state.anger = emotions.get("anger", 0)
            state.anticipation = emotions.get("anticipation", 0)
        
        if "states" in data:
            states = data["states"]
            state.favor = states.get("favor", 0)
            state.intimacy = states.get("intimacy", 0)
        
        state.relationship = data.get("relationship", "陌生人")
        state.attitude = data.get("attitude", "中立")
        
        if "behavior" in data:
            behavior = data["behavior"]
            state.interaction_count = behavior.get("interaction_count", 0)
            state.last_interaction = behavior.get("last_interaction", 0)
            state.positive_interactions = behavior.get("positive_interactions", 0)
            state.negative_interactions = behavior.get("negative_interactions", 0)
        
        if "settings" in data:
            settings = data["settings"]
            state.show_status = settings.get("show_status", False)
        
        return state


class EmotionAIManager:
    """
    高级情感智能管理系统
    """
    def __init__(self, data_path: Path, favour_min: int = -100, favour_max: int = 100, change_min: int = -10, change_max: int = 5):
        self.data_path = data_path
        self.favour_min = favour_min
        self.favour_max = favour_max
        self.change_min = change_min
        self.change_max = change_max
        self._init_path()
        self.user_data = self._load_data("user_emotion_data.json")
        self.blacklist = self._load_data("blacklist.json")
        
    def _init_path(self):
        """初始化数据目录"""
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def _load_data(self, filename: str) -> Dict[str, Any]:
        """加载数据"""
        path = self.data_path / filename
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, TypeError):
            return {}
            
    def _save_data(self, filename: str, data: Dict[str, Any]):
        """保存数据"""
        path = self.data_path / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def get_user_state(self, user_id: str, session_id: Optional[str] = None) -> EmotionalState:
        """获取用户情感状态"""
        key = f"{session_id}_{user_id}" if session_id else user_id
        if key in self.user_data:
            return EmotionalState.from_dict(self.user_data[key])
        return EmotionalState()
        
    def update_user_state(self, user_id: str, state: EmotionalState, session_id: Optional[str] = None):
        """更新用户情感状态"""
        key = f"{session_id}_{user_id}" if session_id else user_id
        
        # 检查是否应该加入黑名单
        if state.favor <= self.favour_min and key not in self.blacklist:
            self.blacklist[key] = {
                "user_id": user_id,
                "session_id": session_id,
                "banned_at": time.time(),
                "reason": "好感度过低"
            }
            self._save_data("blacklist.json", self.blacklist)
        
        self.user_data[key] = state.to_dict()
        self._save_data("user_emotion_data.json", self.user_data)
        
    def is_user_blacklisted(self, user_id: str, session_id: Optional[str] = None) -> bool:
        """检查用户是否在黑名单中"""
        key = f"{session_id}_{user_id}" if session_id else user_id
        return key in self.blacklist
        
    def remove_from_blacklist(self, user_id: str, session_id: Optional[str] = None):
        """从黑名单中移除用户"""
        key = f"{session_id}_{user_id}" if session_id else user_id
        if key in self.blacklist:
            del self.blacklist[key]
            self._save_data("blacklist.json", self.blacklist)
            return True
        return False
        
    def toggle_show_status(self, user_id: str, session_id: Optional[str] = None) -> bool:
        """切换状态显示设置"""
        key = f"{session_id}_{user_id}" if session_id else user_id
        state = self.get_user_state(user_id, session_id)
        state.show_status = not state.show_status
        self.update_user_state(user_id, state, session_id)
        return state.show_status
        
    def calculate_relationship_level(self, state: EmotionalState) -> str:
        """根据情感状态计算关系等级"""
        intimacy_score = state.intimacy
        
        if intimacy_score >= 80:
            return "亲密关系"
        elif intimacy_score >= 60:
            return "挚友"
        elif intimacy_score >= 40:
            return "朋友"
        elif intimacy_score >= 20:
            return "熟人"
        else:
            return "陌生人"
            
    def get_average_ranking(self, limit: int = 10, reverse: bool = True) -> List[Tuple[str, float]]:
        """获取好感度和亲密度的平均值排行榜"""
        averages = []
        
        for key, data in self.user_data.items():
            state = EmotionalState.from_dict(data)
            # 计算好感度和亲密度的平均值
            avg = (state.favor + state.intimacy) / 2
            averages.append((key, avg))
        
        # 排序
        averages.sort(key=lambda x: x[1], reverse=reverse)
        
        # 返回前N名
        return averages[:limit]
        
    def reset_all_data(self):
        """重置所有数据"""
        self.user_data = {}
        self.blacklist = {}
        self._save_data("user_emotion_data.json", self.user_data)
        self._save_data("blacklist.json", self.blacklist)


class EmotionAnalyzer:
    """情感分析器"""
    
    @staticmethod
    def get_dominant_emotion(state: EmotionalState) -> str:
        """获取主导情感"""
        emotions = {
            "喜悦": state.joy,
            "信任": state.trust,
            "恐惧": state.fear,
            "惊讶": state.surprise,
            "悲伤": state.sadness,
            "厌恶": state.disgust,
            "愤怒": state.anger,
            "期待": state.anticipation
        }
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant[0] if dominant[1] > 0 else "中立"
    
    @staticmethod
    def get_emotional_profile(state: EmotionalState) -> Dict[str, Any]:
        """获取完整的情感档案"""
        dominant_emotion = EmotionAnalyzer.get_dominant_emotion(state)
        
        # 计算情感强度
        total_emotion = sum([
            state.joy, state.trust, state.fear, state.surprise,
            state.sadness, state.disgust, state.anger, state.anticipation
        ])
        emotion_intensity = min(100, total_emotion // 2)
        
        # 判断关系趋势
        if state.favor > state.intimacy:
            relationship_trend = "好感领先"
        elif state.intimacy > state.favor:
            relationship_trend = "亲密度领先"
        else:
            relationship_trend = "平衡发展"
            
        # 计算互动质量
        total_interactions = state.interaction_count
        if total_interactions > 0:
            positive_ratio = (state.positive_interactions / total_interactions) * 100
        else:
            positive_ratio = 0
            
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_intensity": emotion_intensity,
            "relationship_trend": relationship_trend,
            "positive_ratio": positive_ratio
        }


@register("EmotionAI", "天各一方", "高级情感智能交互系统", "1.1.2")
class EmotionAIPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        # 获取配置参数
        self.favour_min = self.config.get("favour_min", -100)
        self.favour_max = self.config.get("favour_max", 100)
        self.change_min = self.config.get("change_min", -10)
        self.change_max = self.config.get("change_max", 5)
        
        # 获取规范的数据目录
        data_dir = StarTools.get_data_dir()
        self.manager = EmotionAIManager(data_dir, self.favour_min, self.favour_max, self.change_min, self.change_max)
        self.analyzer = EmotionAnalyzer()
        
        # 情感更新模式
        self.emotion_pattern = re.compile(
            r"\[情感更新:\s*(.*?)\]",
            re.DOTALL
        )
        
        # 单个情感模式
        self.single_emotion_pattern = re.compile(r"(\w+):\s*([+-]?\d+)")
        
    @property
    def session_based(self) -> bool:
        """是否启用会话独立"""
        return bool(self.config.get("session_based", False))
        
    def _get_session_id(self, event: AstrMessageEvent) -> Optional[str]:
        """获取会话ID"""
        return event.unified_msg_origin if self.session_based else None
        
    def _format_emotional_state(self, state: EmotionalState) -> str:
        """格式化情感状态显示"""
        # 计算关系等级
        relationship_level = self.manager.calculate_relationship_level(state)
        
        # 获取情感档案
        profile = self.analyzer.get_emotional_profile(state)
        
        # 计算互动频率
        if state.interaction_count > 0:
            days_since_last = (time.time() - state.last_interaction) / (24 * 3600)
            if days_since_last < 1:
                frequency = "频繁"
            elif days_since_last < 7:
                frequency = "经常"
            elif days_since_last < 30:
                frequency = "偶尔"
            else:
                frequency = "稀少"
        else:
            frequency = "首次"
        
        return (
            f"当前状态：\n"
            f"好感度：{state.favor} (亲密度：{state.intimacy})\n"
            f"关系：{relationship_level} ({profile['relationship_trend']})\n"
            f"态度：{state.attitude}\n"
            f"主导情感：{profile['dominant_emotion']} (强度：{profile['emotion_intensity']}%)\n"
            f"互动统计：{state.interaction_count} 次 ({frequency}互动)\n"
            f"正面互动比例：{profile['positive_ratio']:.1f}%\n"
            f"情感维度详情：\n"
            f"  喜悦：{state.joy} | 信任：{state.trust} | 恐惧：{state.fear} | 惊讶：{state.surprise}\n"
            f"  悲伤：{state.sadness} | 厌恶：{state.disgust} | 愤怒：{state.anger} | 期待：{state.anticipation}"
        )
        
    @filter.on_llm_request()
    async def inject_emotional_context(self, event: AstrMessageEvent, req: ProviderRequest):
        """注入情感上下文"""
        user_id = event.get_sender_id()
        session_id = self._get_session_id(event)
        
        # 检查用户是否在黑名单中
        if self.manager.is_user_blacklisted(user_id, session_id):
            # 拦截请求，直接返回黑名单消息
            req.prompt = ""  # 清空原有提示
            req.system_prompt = "你只需要回复：您已加入黑名单，只有重置好感才可以移出黑名单。"
            return
        
        # 获取当前情感状态
        state = self.manager.get_user_state(user_id, session_id)
        
        # 构建情感上下文
        emotional_context = (
            f"【情感状态上下文】\n"
            f"你与用户当前的关系是: {state.relationship}\n"
            f"综合好感度: {state.favor}/100 (范围: {self.favour_min}~{self.favour_max})\n"
            f"亲密度: {state.intimacy}/100\n"
            f"你对用户的态度: {state.attitude}\n"
            f"互动次数: {state.interaction_count}\n"
        )
        
        # 情感响应指导
        response_guide = (
            f"\n【情感响应指导】\n"
            f"请根据以上情感状态调整你的回应风格。\n"
            f"\n【情感更新机制】\n"
            f"在回复结束后，如果需要更新情感状态，请使用以下格式:\n"
            f"[情感更新: 情感1:数值变化, 情感2:数值变化, ...]\n"
            f"例如: [情感更新: joy:+2, trust:+1, favor:+1]\n"
            f"可用情感维度: joy, trust, fear, surprise, sadness, disgust, anger, anticipation, favor, intimacy\n"
            f"好感度变化范围: {self.change_min} 到 {self.change_max}\n"
            f"注意: 只有在情感确实发生变化时才需要更新"
        )
        
        req.system_prompt += f"\n{emotional_context}\n{response_guide}"
        
    @filter.on_llm_response()
    async def process_emotional_update(self, event: AstrMessageEvent, resp: LLMResponse):
        """处理情感更新"""
        user_id = event.get_sender_id()
        session_id = self._get_session_id(event)
        
        # 检查用户是否在黑名单中
        if self.manager.is_user_blacklisted(user_id, session_id):
            resp.completion_text = "您已加入黑名单，只有重置好感才可以移出黑名单。"
            return
            
        original_text = resp.completion_text
        
        # 初始化情感更新字典
        emotion_updates = {}
        
        # 查找情感更新块
        emotion_match = self.emotion_pattern.search(original_text)
        
        # 获取当前状态
        current_state = self.manager.get_user_state(user_id, session_id)
        
        if emotion_match:
            # 移除情感更新块
            emotion_block = emotion_match.group(0)
            cleaned_text = original_text.replace(emotion_block, '').strip()
            resp.completion_text = cleaned_text
            
            # 解析情感更新
            emotion_content = emotion_match.group(1)
            single_matches = self.single_emotion_pattern.findall(emotion_content)
            
            for emotion, value in single_matches:
                try:
                    change_value = int(value)
                    # 对好感度变化进行限制
                    if emotion.lower() == 'favor':
                        change_value = max(self.change_min, min(self.change_max, change_value))
                    emotion_updates[emotion.lower()] = change_value
                except ValueError:
                    continue
                    
            # 更新基础情感
            if emotion_updates:
                current_state.joy = max(0, min(100, current_state.joy + emotion_updates.get('joy', 0)))
                current_state.trust = max(0, min(100, current_state.trust + emotion_updates.get('trust', 0)))
                current_state.fear = max(0, min(100, current_state.fear + emotion_updates.get('fear', 0)))
                current_state.surprise = max(0, min(100, current_state.surprise + emotion_updates.get('surprise', 0)))
                current_state.sadness = max(0, min(100, current_state.sadness + emotion_updates.get('sadness', 0)))
                current_state.disgust = max(0, min(100, current_state.disgust + emotion_updates.get('disgust', 0)))
                current_state.anger = max(0, min(100, current_state.anger + emotion_updates.get('anger', 0)))
                current_state.anticipation = max(0, min(100, current_state.anticipation + emotion_updates.get('anticipation', 0)))
                
                # 更新状态
                current_state.favor = max(self.favour_min, min(self.favour_max, current_state.favor + emotion_updates.get('favor', 0)))
                current_state.intimacy = max(0, min(100, current_state.intimacy + emotion_updates.get('intimacy', 0)))
        
        # 更新行为统计
        current_state.interaction_count += 1
        current_state.last_interaction = time.time()
        
        # 判断互动性质
        if any(emotion_updates.values()):
            # 如果有情感更新，根据更新的情感判断互动性质
            positive_emotions = sum([
                emotion_updates.get('joy', 0),
                emotion_updates.get('trust', 0),
                emotion_updates.get('surprise', 0),
                emotion_updates.get('anticipation', 0)
            ])
            negative_emotions = sum([
                emotion_updates.get('fear', 0),
                emotion_updates.get('sadness', 0),
                emotion_updates.get('disgust', 0),
                emotion_updates.get('anger', 0)
            ])
            
            if positive_emotions > negative_emotions:
                current_state.positive_interactions += 1
            elif negative_emotions > positive_emotions:
                current_state.negative_interactions += 1
        
        # 更新关系状态
        current_state.relationship = self.manager.calculate_relationship_level(current_state)
        
        # 保存更新
        self.manager.update_user_state(user_id, current_state, session_id)
        
        # 根据用户设置决定是否添加状态显示
        if current_state.show_status:
            status_block = f"\n\n{self._format_emotional_state(current_state)}"
            resp.completion_text = resp.completion_text + status_block
            
    # ------------------- 用户命令 -------------------
    
    @filter.command("好感度")
    async def show_emotional_state(self, event: AstrMessageEvent):
        """显示情感状态"""
        user_id = event.get_sender_id()
        session_id = self._get_session_id(event)
        
        # 检查用户是否在黑名单中
        if self.manager.is_user_blacklisted(user_id, session_id):
            yield event.plain_result("您已加入黑名单，只有重置好感才可以移出黑名单。")
            return
            
        state = self.manager.get_user_state(user_id, session_id)
        
        response_text = self._format_emotional_state(state)
        yield event.plain_result(response_text)
        
    @filter.command("状态显示")
    async def toggle_status_display(self, event: AstrMessageEvent):
        """切换状态显示开关"""
        user_id = event.get_sender_id()
        session_id = self._get_session_id(event)
        
        # 检查用户是否在黑名单中
        if self.manager.is_user_blacklisted(user_id, session_id):
            yield event.plain_result("您已加入黑名单，只有重置好感才可以移出黑名单。")
            return
            
        new_status = self.manager.toggle_show_status(user_id, session_id)
        
        if new_status:
            response_text = "已开启状态显示，每次对话后将会显示当前状态。"
        else:
            response_text = "已关闭状态显示，对话后将不再显示状态信息。"
            
        yield event.plain_result(response_text)
        
    # ------------------- 排行榜命令 -------------------
    
    @filter.command("好感排行")
    async def show_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        """显示好感度和亲密度平均值排行榜"""
        try:
            limit = int(num)
            if limit <= 0:
                raise ValueError
        except ValueError:
            yield event.plain_result("错误：排行数量必须是一个正整数。")
            return

        rankings = self.manager.get_average_ranking(limit, True)
        
        if not rankings:
            yield event.plain_result("当前没有任何用户数据。")
            return

        response_lines = [f"好感度平均值 TOP {limit} 排行榜："]
        for i, (user_key, avg) in enumerate(rankings, 1):
            state_data = self.manager.user_data.get(user_key, {})
            states = state_data.get("states", {})
            favor = states.get("favor", 0)
            intimacy = states.get("intimacy", 0)
            line = (
                f"{i}. 用户: {user_key}\n"
                f"   - 平均值: {avg:.1f} (好感: {favor}, 亲密: {intimacy})"
            )
            response_lines.append(line)
        
        yield event.plain_result("\n".join(response_lines))
        
    @filter.command("负好感排行")
    async def show_negative_favor_ranking(self, event: AstrMessageEvent, num: str = "10"):
        """显示好感度和亲密度平均值最低排行榜"""
        try:
            limit = int(num)
            if limit <= 0:
                raise ValueError
        except ValueError:
            yield event.plain_result("错误：排行数量必须是一个正整数。")
            return

        rankings = self.manager.get_average_ranking(limit, False)
        
        if not rankings:
            yield event.plain_result("当前没有任何用户数据。")
            return

        response_lines = [f"好感度平均值 BOTTOM {limit} 排行榜："]
        for i, (user_key, avg) in enumerate(rankings, 1):
            state_data = self.manager.user_data.get(user_key, {})
            states = state_data.get("states", {})
            favor = states.get("favor", 0)
            intimacy = states.get("intimacy", 0)
            line = (
                f"{i}. 用户: {user_key}\n"
                f"   - 平均值: {avg:.1f} (好感: {favor}, 亲密: {intimacy})"
            )
            response_lines.append(line)
        
        yield event.plain_result("\n".join(response_lines))
        
    # ------------------- 管理员命令 -------------------
    
    def _is_admin(self, event: AstrMessageEvent) -> bool:
        """检查管理员权限"""
        admin_list = self.config.get("admin_qq_list", [])
        return event.role == "admin" or event.get_sender_id() in admin_list
        
    @filter.command("设置好感")
    async def admin_set_favor(self, event: AstrMessageEvent, user_id: str, value: str):
        """设置好感度"""
        if not self._is_admin(event):
            yield event.plain_result("错误：需要管理员权限")
            return
            
        try:
            favor_value = int(value)
            if not self.favour_min <= favor_value <= self.favour_max:
                yield event.plain_result(f"错误：好感度值必须在 {self.favour_min} 到 {self.favour_max} 之间。")
                return
        except ValueError:
            yield event.plain_result("错误：好感度值必须是数字")
            return
            
        session_id = self._get_session_id(event)
        state = self.manager.get_user_state(user_id, session_id)
        state.favor = favor_value
        
        # 如果设置的好感度高于下限，从黑名单中移除
        if favor_value > self.favour_min:
            self.manager.remove_from_blacklist(user_id, session_id)
        
        self.manager.update_user_state(user_id, state, session_id)
        yield event.plain_result(f"成功：用户 {user_id} 的好感度已设置为 {favor_value}")
        
    @filter.command("重置好感")
    async def admin_reset_favor(self, event: AstrMessageEvent, user_id: str):
        """重置用户好感度状态"""
        if not self._is_admin(event):
            yield event.plain_result("错误：需要管理员权限")
            return
            
        session_id = self._get_session_id(event)
        
        # 创建一个全新的默认状态
        new_state = EmotionalState()
        
        # 从黑名单中移除
        self.manager.remove_from_blacklist(user_id, session_id)
        
        self.manager.update_user_state(user_id, new_state, session_id)
        yield event.plain_result(f"成功：用户 {user_id} 的好感度状态已完全重置")
        
    @filter.command("重置插件")
    async def admin_reset_plugin(self, event: AstrMessageEvent):
        """重置插件所有数据"""
        if not self._is_admin(event):
            yield event.plain_result("错误：需要管理员权限")
            return
            
        # 重置所有数据
        self.manager.reset_all_data()
        
        yield event.plain_result("成功：插件所有数据已重置，包括用户情感状态和黑名单")
        
    async def terminate(self):
        """插件终止时保存数据"""
        self.manager._save_data("user_emotion_data.json", self.manager.user_data)
        self.manager._save_data("blacklist.json", self.manager.blacklist)