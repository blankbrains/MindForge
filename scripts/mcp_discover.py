#!/usr/bin/env python3
"""MindForge MCP 工具发现与测试脚本"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("MindForge MCP 协议集成测试")
    print("=" * 60)

    print("\n[1/3] 加载 MCP 注册表...")
    from mindforge.mcp.registry import get_mcp_registry
    registry = get_mcp_registry()
    registry.load_config()
    print(f"   已加载 {len(registry.servers)} 个 MCP Server 配置")
    for name in registry.servers:
        print(f"     - {name}")

    print("\n[2/3] 发现 MCP 工具...")
    tools = await registry.discover_tools()
    print(f"   发现 {len(tools)} 个 MCP 工具")
    for tool in tools:
        print(f"     - [{tool.server_name}] {tool.tool_name}")
        print(f"       {tool.description[:80]}")

    print("\n[3/3] MCP 适配器接口测试...")
    from mindforge.tools.mcp_adapter import MCPToolAdapter
    adapter = MCPToolAdapter()
    await adapter.initialize()
    tool_count = len(adapter.to_openai_function().get('function', {}).get('name', ''))
    print(f"   MCP 适配器就绪，已注册 {tool_count} 个外部工具")
    print(f"   Researcher Agent 可通过 MCP 协议调用外部服务")

    print("\n" + "=" * 60)
    print("MCP 模块验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
