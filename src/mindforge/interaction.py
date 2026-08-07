"""Deterministic routing for bounded conversational turns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ConversationKind = Literal[
    "greeting",
    "acknowledgement",
    "reaction",
    "thanks",
    "farewell",
    "presence",
    "status",
    "identity",
    "apology",
]


@dataclass(frozen=True)
class ConversationalTurn:
    """A bounded social turn that does not require the research pipeline."""

    kind: ConversationKind
    response: str


_EDGE_PUNCTUATION_RE = re.compile(
    r"^[\s,，.。!！?？~～…·、;；:：\"'“”‘’()（）\[\]【】{}<>《》]+"
    r"|[\s,，.。!！?？~～…·、;；:：\"'“”‘’()（）\[\]【】{}<>《》]+$"
)
_REACTION_RE = re.compile(
    r"^(?:哈{1,12}|嘿{2,12}|呵{2,12}|嘻{2,12}|"
    r"笑死(?:我了)?|绷不住了|乐了|6{2,8})$"
)
_ACKNOWLEDGEMENT_RE = re.compile(
    r"^(?:[嗯唔哦噢喔啊]{1,6}|"
    r"好(?:的|吧|嘞|呀|啊)?|行(?:吧|啊|呀)?|可以|"
    r"知道了|明白了|明白|懂了|收到|没问题|"
    r"ok(?:ay)?|got it|fine)$"
)

_GREETINGS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "你好",
        "你好啊",
        "你好呀",
        "您好",
        "嗨",
        "哈喽",
        "早上好",
        "上午好",
        "下午好",
        "晚上好",
    }
)
_THANKS = frozenset({"谢谢", "多谢", "感谢", "谢了", "thanks", "thank you"})
_FAREWELLS = frozenset(
    {"再见", "拜拜", "回头见", "晚安", "bye", "goodbye"}
)
_PRESENCE = frozenset(
    {"在吗", "在不在", "你在吗", "有人吗", "干嘛呢", "干啥呢"}
)
_STATUS = frozenset(
    {"咋了", "怎么了", "咋回事", "什么情况", "干嘛", "干啥"}
)
_IDENTITY = frozenset(
    {
        "你是谁",
        "你叫什么",
        "你叫什么名字",
        "你是干什么的",
        "你能干什么",
        "你能干啥",
    }
)
_APOLOGIES = frozenset({"抱歉", "对不起", "不好意思", "sorry"})

_RESPONSES: dict[ConversationKind, str] = {
    "greeting": "你好，我在。你想聊什么？",
    "acknowledgement": "好，明白了。",
    "reaction": "哈哈。",
    "thanks": "不客气。",
    "farewell": "再见。",
    "presence": "在，怎么了？",
    "status": "我在。你是想继续刚才的话题，还是遇到了什么问题？",
    "identity": (
        "我是 MindForge，一个支持连续对话、知识检索和多 Agent 研究的助手。"
    ),
    "apology": "没关系。",
}


def normalize_interaction_text(text: str) -> str:
    """Normalize only surface form; never rewrite substantive user content."""
    normalized = " ".join(text.strip().casefold().split())
    previous = None
    while normalized != previous:
        previous = normalized
        normalized = _EDGE_PUNCTUATION_RE.sub("", normalized)
    return normalized


def classify_conversational_turn(text: str) -> ConversationalTurn | None:
    """Classify short, self-contained social turns using anchored rules."""
    normalized = normalize_interaction_text(text)
    if not normalized or len(normalized) > 24:
        return None

    kind: ConversationKind | None = None
    if normalized in _GREETINGS:
        kind = "greeting"
    elif normalized in _THANKS:
        kind = "thanks"
    elif normalized in _FAREWELLS:
        kind = "farewell"
    elif normalized in _PRESENCE:
        kind = "presence"
    elif normalized in _STATUS:
        kind = "status"
    elif normalized in _IDENTITY:
        kind = "identity"
    elif normalized in _APOLOGIES:
        kind = "apology"
    elif _REACTION_RE.fullmatch(normalized):
        kind = "reaction"
    elif _ACKNOWLEDGEMENT_RE.fullmatch(normalized):
        kind = "acknowledgement"

    if kind is None:
        return None
    return ConversationalTurn(kind=kind, response=_RESPONSES[kind])


def is_conversational_task(text: str) -> bool:
    """Return whether a turn can bypass research and model execution."""
    return classify_conversational_turn(text) is not None
