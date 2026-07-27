#!/usr/bin/env python3
"""MCP 工具发现与调用演示脚本 — 展示 MindForge 的 MCP 协议集成能力"""

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def safe_terminal_text(value: object, limit: int) -> str:
    text = re.sub(
        r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]",
        "",
        str(value),
    )
    return text[:limit]


async def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("MindForge — MCP 协议集成演示")
    print("=" * 60)

    # Step 1: Load MCP configuration
    print("\n[1/5] 加载 MCP 配置...")
    from mindforge.config import get_settings

    settings = get_settings()
    config_json = settings.mcp.mcp_servers_json.strip()
    if not config_json:
        print("   未在根目录 .env 的 MCP_MCP_SERVERS_JSON 中配置 MCP Server")
        return

    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as exc:
        print(f"   MCP_MCP_SERVERS_JSON 格式错误: {exc}")
        return

    servers = config.get("mcpServers", {})
    print(f"   发现 {len(servers)} 个 MCP 服务器配置:")
    for name, cfg in servers.items():
        print(
            f"      - {safe_terminal_text(name, 100)}: "
            f"{safe_terminal_text(cfg.get('command', ''), 200)} "
            f"({len(cfg.get('args', []))} args)"
        )

    # Step 2: Test MCP Registry initialization
    print("\n[2/5] 初始化 MCP Registry...")
    from mindforge.mcp.registry import MCPRegistry

    registry = MCPRegistry()
    try:
        registry.load_config_json(config_json)
        print(f"   加载完成: {len(registry.servers)} 个服务器")
    except Exception as e:
        print(f"   加载失败: {e}")
        print("   (非关键错误，继续演示...)")

    try:
        await registry.start_all(timeout=settings.mcp.mcp_tool_timeout)
        tools = await registry.discover_all_tools()

        # Step 3: Test discovered MCP tools
        print("\n[3/5] 测试 MCP 工具发现...")
        if tools:
            print(f"   可用工具 ({len(tools)}):")
            for tool in tools:
                print(
                    f"      - {safe_terminal_text(tool.name, 100)}: "
                    f"{safe_terminal_text(tool.description, 60)}"
                )
        else:
            print("   未发现 MCP 工具 (MCP 服务器可能未运行)")
            print("   提示: MCP 服务器通过 npx/uvx 按需启动，需要 Node.js/Python 环境")

        # Step 4: Verify OpenAI function format conversion
        print("\n[4/5] 验证 OpenAI Function Calling 格式转换...")
        functions = registry.get_openai_tools()
        print(f"   转换 {len(functions)} 个函数:")
        for func in functions:
            name = func.get("function", {}).get("name", "unknown")
            print(f"      - {safe_terminal_text(name, 100)}")
        if functions:
            print("   ✅ Function Calling 格式转换正常")
        else:
            print("   ⚠️ 无可用工具 (MCP 服务器需先连接)")

        # Step 5: Integration — MCP in the Agent pipeline
        print("\n[5/5] MCP 集成验证...")
        print("""
   当 MCP 服务器正常运行后，Researcher Agent 的流程:

   ┌─ Researcher Agent ────────────────────────────┐
   │  1. 接收子任务                                  │
   │  2. 选择工具:                                   │
   │     ├── RAGTool (知识库检索)                    │
   │     ├── WebSearchTool (网络搜索)                │
   │     ├── MCPToolAdapter ──── 动态 MCP 工具      │
   │     │       ├── context7 → 库文档查询           │
   │     │       ├── github → 代码仓库操作           │
   │     │       └── qdrant → 向量库管理            │
   │  3. ReAct 循环: 思考 → 执行 → 观察              │
   └───────────────────────────────────────────────┘

   MCP 协议优势:
   • 工具标准化: 所有外部工具通过统一 JSON-RPC 协议接入
   • 动态发现: 启动时读取根目录 .env，无需额外配置文件
   • 运行时注册: 支持动态添加/移除工具
   • 自动适配: MCP 工具自动转为 OpenAI Function Calling 格式
   """)
        print("=" * 60)
        print("演示完成")
        print("=" * 60)
    finally:
        await registry.stop_all()


if __name__ == "__main__":
    asyncio.run(main())
