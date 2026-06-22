#!/usr/bin/env python3
"""MCP 工具发现与调用演示脚本 — 展示 MindForge 的 MCP 协议集成能力"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("MindForge — MCP 协议集成演示")
    print("=" * 60)

    # Step 1: Load MCP configuration
    print("\n[1/5] 加载 MCP 配置...")
    mcp_config_path = Path(__file__).parent.parent / "mcp.json"
    if not mcp_config_path.exists():
        print(f"   ❌ 未找到 MCP 配置文件: {mcp_config_path}")
        return

    with open(mcp_config_path) as f:
        config = json.load(f)

    servers = config.get("mcpServers", {})
    print(f"   发现 {len(servers)} 个 MCP 服务器配置:")
    for name, cfg in servers.items():
        print(f"      - {name}: {cfg.get('command')} {' '.join(cfg.get('args', []))}")

    # Step 2: Test MCP Registry initialization
    print("\n[2/5] 初始化 MCP Registry...")
    from mindforge.mcp.registry import MCPRegistry

    registry = MCPRegistry()
    try:
        registry.load_config(str(mcp_config_path))
        print(f"   加载完成: {len(registry.servers)} 个服务器")
    except Exception as e:
        print(f"   加载失败: {e}")
        print("   (非关键错误，继续演示...)")

    # Step 3: Test MCPToolAdapter
    print("\n[3/5] 测试 MCPToolAdapter...")
    try:
        from mindforge.tools.mcp_adapter import MCPToolAdapter

        adapter = MCPToolAdapter(config_path=str(mcp_config_path))
        available = adapter.list_available_tools()
        if available:
            print(f"   可用工具 ({len(available)}):")
            for t in available:
                print(f"      - {t['name']}: {t.get('description', '无描述')[:60]}")
        else:
            print("   未发现 MCP 工具 (MCP 服务器可能未运行)")
            print("   提示: MCP 服务器通过 npx/uvx 按需启动，需要 Node.js/Python 环境")
    except Exception as e:
        print(f"   MCPToolAdapter 初始化: {e}")

    # Step 4: Verify OpenAI function format conversion
    print("\n[4/5] 验证 OpenAI Function Calling 格式转换...")
    try:
        adapter = MCPToolAdapter(config_path=str(mcp_config_path))
        functions = adapter.to_openai_functions()
        print(f"   转换 {len(functions)} 个函数:")
        for func in functions:
            name = func.get("function", {}).get("name", "unknown")
            print(f"      - {name}")
        if functions:
            print("   ✅ Function Calling 格式转换正常")
        else:
            print("   ⚠️ 无可用工具 (MCP 服务器需先连接)")
    except Exception as e:
        print(f"   ❌ 格式转换失败: {e}")

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
   • 动态发现: 启动时自动扫描 mcp.json，无需硬编码
   • 运行时注册: 支持动态添加/移除工具
   • 自动适配: MCP 工具自动转为 OpenAI Function Calling 格式
   """)
    print("=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
