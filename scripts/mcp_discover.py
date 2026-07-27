#!/usr/bin/env python3
"""MindForge MCP 工具发现与测试脚本"""
import asyncio
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def safe_terminal_text(value: object, limit: int) -> str:
    return re.sub(
        r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]",
        "",
        str(value),
    )[:limit]


async def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("MindForge MCP 协议集成测试")
    print("=" * 60)

    print("\n[1/3] 加载 MCP 注册表...")
    from mindforge.config import get_settings
    from mindforge.mcp.registry import get_mcp_registry

    config_json = get_settings().mcp.mcp_servers_json.strip()
    if not config_json:
        print("   根目录 .env 中未配置 MCP_MCP_SERVERS_JSON")
        return
    registry = get_mcp_registry()
    registry.load_config_json(config_json)
    print(f"   已加载 {len(registry.servers)} 个 MCP Server 配置")
    for name in registry.servers:
        print(f"     - {name}")

    try:
        print("\n[2/3] 发现 MCP 工具...")
        await registry.start_all(
            timeout=get_settings().mcp.mcp_tool_timeout
        )
        tools = await registry.discover_all_tools()
        print(f"   发现 {len(tools)} 个 MCP 工具")
        for tool in tools:
            print(
                f"     - [{safe_terminal_text(tool.server_name, 100)}] "
                f"{safe_terminal_text(tool.name, 100)}"
            )
            print(f"       {safe_terminal_text(tool.description, 80)}")

        print("\n[3/3] MCP 适配器接口测试...")
        from mindforge.tools.mcp_adapter import MCPToolAdapter
        adapter = MCPToolAdapter()
        openai_tools = await adapter.discover_openai_tools()
        tool_count = len(openai_tools)
        print(f"   MCP 适配器就绪，已注册 {tool_count} 个外部工具")
        print("   Researcher Agent 可通过 MCP 协议调用外部服务")
    finally:
        await registry.stop_all()

    print("\n" + "=" * 60)
    print("MCP 模块验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
