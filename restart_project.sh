#!/bin/bash

# 防止脚本并发运行
# 如果上一个脚本还没跑完，当前脚本直接退出
exec 9<"$0"
flock -n 9 || exit 1
# --- 配置 ---
PROJECT_DIR="/home/hk/PAE/AdverShield"
FRP_DIR="/home/hk/frp_0.67.0_linux_amd64"
LOG_FILE="$PROJECT_DIR/monitor.log"
PYTHON_MAIL_SCRIPT="$PROJECT_DIR/send_mail.py"
CONDA_PATH="~/anaconda3/etc/profile.d/conda.sh"

send_notification() {
    local module_name=$1
    local status_code=$2
    local extra_msg=$3
    if [ $status_code -eq 0 ]; then
        subject="【成功】$module_name"
        content="时间: $(date '+%Y-%m-%d %H:%M:%S')\n$module_name 已恢复。"
    else
        subject="【失败】$module_name"
        content="时间: $(date '+%Y-%m-%d %H:%M:%S')\n错误原因: $extra_msg"
    fi
    python3 "$PYTHON_MAIL_SCRIPT" "$subject" "$content" >> $LOG_FILE 2>&1
}

# 1. 基础进程守护 (frpc)
if ! pgrep -x "frpc" > /dev/null; then
    tmux kill-session -t frp_client 2>/dev/null
    tmux new-session -d -s frp_client "cd $FRP_DIR && ./frpc -c frpc.toml"
    send_notification "frp_client" $? "进程掉线自动重启"
fi

# 2. 凌晨 03:00 核心重启逻辑
CURRENT_TIME=$(date +%H:%M)
if [ "$CURRENT_TIME" == "03:00" ]; then
    echo "[$(date)] 开始深度维护..." >> $LOG_FILE

    # --- 第一步：彻底清理旧进程 ---
    docker stop laughing_gould 2>/dev/null
    # 强制杀死所有可能占用显卡的残留 python/carla 进程
    fuser -kv /dev/nvidia* 2>/dev/null 
    sleep 5 

    # --- 第二步：启动 Docker 并增加状态判定 ---
    docker start laughing_gould
    sleep 10 # 关键：给 Carla 留出分配显存的时间

    # 检查容器是否真的在运行
    IS_RUNNING=$(docker inspect -f '{{.State.Running}}' laughing_gould)
    if [ "$IS_RUNNING" == "true" ]; then
        send_notification "Docker(laughing_gould)" 0
    else
        # 抓取最后一行报错发给邮件
        ERR_LOG=$(docker logs --tail 1 laughing_gould)
        send_notification "Docker(laughing_gould)" 1 "容器启动后崩溃，报错：$ERR_LOG"
        exit 1 # 容器挂了就没必要执行后面了
    fi

    # --- 第三步：清理端口并启动 Tmux ---
    for port in 8000 7100; do
        PID=$(lsof -t -i:$port)
        [ ! -z "$PID" ] && kill -9 $PID
    done

    tmux kill-session -t advershield_project 2>/dev/null
    sleep 2
    
    # 启动应用
    tmux new-session -d -s advershield_project -n "carla_server"
    tmux send-keys -t advershield_project:0 "source $CONDA_PATH && conda activate carla_37 && cd $PROJECT_DIR && python carla_server.py" C-m
    sleep 5
    tmux new-window -t advershield_project -n "main_app"
    tmux send-keys -t advershield_project:1 "source $CONDA_PATH && conda activate AdverShield && cd $PROJECT_DIR && python main.py" C-m

    send_notification "核心服务(Tmux)" $? "项目会话重启完毕"
fi