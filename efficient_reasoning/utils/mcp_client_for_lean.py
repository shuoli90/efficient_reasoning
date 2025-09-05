# import asyncio
# from typing import Optional
# from contextlib import AsyncExitStack

# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client

# from dotenv import load_dotenv

# load_dotenv()  # load environment variables from .env

# class MCPClient:
#     def __init__(self):
#         # Initialize session and client objects
#         self.session: Optional[ClientSession] = None
#         self.exit_stack = AsyncExitStack()
#     # methods will go here
    
#     async def connect_to_server(self):
#         """Connect to an MCP server

#         Args:
#             server_script_path: Path to the server script (.py or .js)
#         """

#         command = "/home/sanupam/miniconda3/bin/conda"
#         server_params = StdioServerParameters(
#             command=command,
#             args=[
#                 "run",
#                 "-n",
#                 "leanmcpenv",
#                 "python"
#                 "-m", 
#                 "lean_docker_mcp"],
#             env=None
#         )

#         stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
#         self.stdio, self.write = stdio_transport
#         self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

#         await self.session.initialize()

#         # List available tools
#         response = await self.session.list_tools()
#         tools = response.tools
#         print("\nConnected to server with tools:", [tool.name for tool in tools])
    
#     async def cleanup(self):
#         """Clean up resources"""
#         await self.exit_stack.aclose()
    
# async def main():

#     client = MCPClient()
#     try:
#         await client.connect_to_server()
#     finally:
#         await client.cleanup()

import tempfile
import os
import subprocess
import signal

if __name__ == "__main__":
    # import sys
    # asyncio.run(main())
    with open("lean_eval_dir/lake_test_2.lean", 'r') as f:
    #with open("lean_eval_dir/tmpb_7mg0wn.lean", 'r') as f:
        candidate = f.read()
    tmp_file_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean_eval_dir")
    old_tmp_dir = tempfile.gettempdir()
    tempfile.tempdir = tmp_file_directory
    tmp_lean_file = tempfile.NamedTemporaryFile(suffix=".lean")
    with open(tmp_lean_file.name, 'w') as f:
        f.write(candidate)
    cmd = f"lake lean {tmp_lean_file.name}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd = tmp_file_directory, stderr=subprocess.PIPE)
    try:
        stdout, _ = process.communicate()
        print(f"Stdout: {stdout} \n")
        print(f"Stderr: {_} \n")
        error = stdout.decode()
        print(f"Decoded Error: {error} \n")
    except:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        print("Unexpected error")