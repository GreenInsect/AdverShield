#!/bin/bash

# --- 配置 ---
CONTAINER_NAME="laughing_gould"
PROJECT_DIR="/home/hk/PAE/AdverShield"
LOG_FILE="$PROJECT_DIR/container_watchdog.log"
PYTHON_MAIL_SCRIPT="$PROJECT_DIR/send_mail.py"

# --- 邮件通知函数 ---
send_alert() {
    local subject="【告警】容器 $CONTAINER_NAME 异常停止"
    local content="检测到容器 $CONTAINER_NAME 已关闭。尝试自动重启...\n时间: $(date '+%Y-%m-%d %H:%M:%S')\n最后日志: $1"
    python3 "$PYTHON_MAIL_SCRIPT" "$subject" "$content" >> $LOG_FILE 2>&1
}

# 检查容器是否正在运行
IS_RUNNING=$(docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null)

if [ "$IS_RUNNING" != "true" ]; then
    echo "[$(date)] 检测到容器 $CONTAINER_NAME 停止，准备重启..." >> $LOG_FILE
    
    # 获取崩溃瞬间的最后两行日志，用于邮件分析
    LAST_LOG=$(docker logs --tail 2 $CONTAINER_NAME 2>/dev/null)
    
    # 发送告警邮件
    send_alert "$LAST_LOG"
    
    # 尝试暴力清理显存（防止因段错误 Exit 139 导致的无法启动）
    # sudo fuser -kv /dev/nvidia* 2>/dev/null
    sleep 2
    
    # 重新启动容器
    docker start $CONTAINER_NAME
    
    # 二次验证启动结果
    sleep 5
    FINAL_CHECK=$(docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null)
    if [ "$FINAL_CHECK" == "true" ]; then
        echo "[$(date)] 容器已成功拉起。" >> $LOG_FILE
    else
        echo "[$(date)] 自动重启失败，请人工介入！" >> $LOG_FILE
    fi
else
    # 容器运行正常，保持静默
    exit 0
fi