#!/usr/bin/env python3
"""
ASR API Client Example

Demonstrates how to call the ASR API to get task results.
This script can run alongside the API server to receive task classifications.
"""

import requests
import time
import json
import logging
from typing import Optional, Dict, Any

from apis import (
    control_robot,
    replay,
    hold_phone,
)
from lerobot.common.robot_devices.control_utils import (
    go_to_rest_pos,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ASRAPIClient:
    """Client for the ASR Task Classification API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.robot, self.cfg = control_robot()  # Initialize robot and config
    
    def get_task(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Get the next task result (blocking)"""
        try:
            response = self.session.get(
                f"{self.base_url}/task",
                params={"timeout": timeout},
                timeout=timeout + 5  # Add buffer for HTTP timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 408:
                logger.debug("Timeout waiting for task")
                return None
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.debug("Request timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get ASR system statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get stats: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get stats: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check if the API server is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
            return False
        except requests.exceptions.RequestException:
            return False
    
    def stream_tasks(self, callback):
        """Stream tasks using Server-Sent Events"""
        try:
            response = self.session.get(
                f"{self.base_url}/task/stream",
                stream=True,
                timeout=None
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to start stream: {response.status_code}")
                return
            
            logger.info("Started task stream")
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        callback(data, self.robot, self.cfg)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse stream data: {e}")
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Stream failed: {e}")

def task_handler(task_data: Dict[str, Any], robot, cfg):
    """Handle received task data"""
    if "error" in task_data:
        logger.error(f"Stream error: {task_data['error']}")
        return
    
    task = task_data.get("task")
    original_text = task_data.get("original_text")
    timestamp = task_data.get("timestamp")
    
    print(f"\n=== 收到任务 ===")
    print(f"任务: {task}")
    print(f"原始文本: {original_text}")
    print(f"时间戳: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
    
    # Execute task-specific logic
    if task == "拿纸巾":
        replay(robot, cfg.control, path="data/napkin.pt")
        # Add your robot control code here
    elif task == "帮忙拿手机":
        hold_phone(robot, cfg.control)
        # Add your robot control code here
    elif task == "松开并归还手机":
        go_to_rest_pos(robot)
        # Add your robot control code here
    elif task == "无":
        print("执行动作: 无相关任务")
    else:
        print(f"执行动作: 未知任务 - {task}")
    
    print("=" * 20)

def main():
    """Main function demonstrating different ways to use the API"""
    client = ASRAPIClient()
    
    # Check if server is healthy
    if not client.health_check():
        logger.error("API server is not healthy. Please start the server first.")
        print("\n请先启动API服务器:")
        print("python3 asr_api_server.py")
        return
    
    logger.info("API server is healthy")
    
    # Show current stats
    stats = client.get_stats()
    if stats:
        print(f"\n=== 系统状态 ===")
        print(f"API状态: {stats.get('api_status')}")
        print(f"队列大小: {stats.get('queue_size')}")
        print(f"音频处理: {stats.get('audio_chunks_processed', 0)} 块")
        print(f"语音识别: {stats.get('recognition_count', 0)} 次")
        print("=" * 20)
    
    try:
        client.stream_tasks(task_handler)
    
    except KeyboardInterrupt:
        print("\n\n程序已退出")
    except Exception as e:
        logger.error(f"程序错误: {e}")

if __name__ == "__main__":
    robot, cfg = control_robot()
    main()