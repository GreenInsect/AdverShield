#!/bin/bash

# --- 配置 ---
PROJECT_DIR="/home/hk/PAE/AdverShield"
FRP_DIR="/home/hk/frp_0.67.0_linux_amd64"
LOG_FILE="$PROJECT_DIR/monitor.log"
PYTHON_MAIL_SCRIPT="$PROJECT_DIR/send_mail.py" # 请确保路径正确
CONDA_PATH="~/anaconda3/etc/profile.d/conda.sh"

# --- 邮件预警函数 ---
# 参数1: 模块名称, 参数2: 执行结果 (0为成功, 其他为失败)
send_notification() {
    local module_name=$1
    local status_code=$2
    local subject=""
    local content=""

    if [ $status_code -eq 0 ]; then
        subject="【成功】服务维护通知: $module_name"
        content="时间: $(date '+%Y-%m-%d %H:%M:%S')\n状态: $module_name 已成功启动/重启并恢复运行。"
    else
        subject="【失败】服务维护报警: $module_name"
        content="时间: $(date '+%Y-%m-%d %H:%M:%S')\n状态: $module_name 重启过程中出现异常, 请人工检查！"
    fi

    # 调用你的 Python 脚本
    python3 "$PYTHON_MAIL_SCRIPT" "$subject" "$content" >> $LOG_FILE 2>&1
}

echo "------------------------------------------" >> $LOG_FILE
echo "[$(date)] 启动环境检查..." >> $LOG_FILE

# --- 1. 检查并维护 frpc (日常进程守护) ---
if pgrep -x "frpc" > /dev/null; then
    echo "frpc 正在运行, 跳过。" >> $LOG_FILE
else
    echo "警告：检测到 frpc 已掉线, 正在重新启动..." >> $LOG_FILE
    tmux kill-session -t frp_client 2>/dev/null
    tmux new-session -d -s frp_client "cd $FRP_DIR && ./frpc -c frpc.toml"
    send_notification "frp_client (自动重连)" $?
fi

# --- 2. 凌晨 03:00 的强制重启逻辑 ---
# 获取当前小时和分钟, 确保只在 03:00 这一分钟内执行
CURRENT_TIME=$(date +%H:%M)

if [ "$CURRENT_TIME" == "03:00" ]; then
    echo "触发凌晨 03:00 定时维护..." >> $LOG_FILE
    
    # A. 停止并重启 Docker
    docker stop laughing_gould 2>/dev/null
    docker start laughing_gould
    send_notification "Docker容器(laughing_gould)" $?

    # B. 清理端口占用
    for port in 8000 7100; do
        PID=$(lsof -t -i:$port)
        if [ ! -z "$PID" ]; then
            kill -9 $PID && echo "清理端口 $port" >> $LOG_FILE
        fi
    done

    # C. 重启 Tmux 核心项目会话
    tmux kill-session -t advershield_project 2>/dev/null
    sleep 2
    
    # 启动 carla_server
    tmux new-session -d -s advershield_project -n "carla_server"
    tmux send-keys -t advershield_project:0 "source $CONDA_PATH && conda activate carla_37 && cd $PROJECT_DIR && python carla_server.py" C-m
    
    # 等待几秒启动 main_app
    sleep 5
    tmux new-window -t advershield_project -n "main_app"
    tmux send-keys -t advershield_project:1 "source $CONDA_PATH && conda activate AdverShield && cd $PROJECT_DIR && python main.py" C-m
    
    # 检查 tmux 会话是否创建成功作为判定
    tmux has-session -t advershield_project 2>/dev/null
    send_notification "核心服务(Carla & MainApp)" $?

    echo "核心服务已完成每日例行重启。" >> $LOG_FILE
else
    echo "当前时间 $CURRENT_TIME, 非 03:00 维护时间。" >> $LOG_FILE
fi