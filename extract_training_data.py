#!/usr/bin/env python3
"""
从训练日志中提取真实的quad_s0和Q_q数据
用于重现QP solver错误
"""

import sys
import os
import torch
import numpy as np
import re

def extract_data_from_logs(log_content):
    """从日志内容中提取quad_s0和Q_q数据"""
    
    # 提取quad_s0数据的正则表达式
    quad_s0_pattern = r'quad_s0 is:\s*tensor\(\[\[(.*?)\]\]'
    
    # 提取Q_q数据的正则表达式
    Q_q_pattern = r'Q_q is:\s*tensor\(\[\[(.*?)\]\]'
    
    quad_s0_matches = re.findall(quad_s0_pattern, log_content, re.DOTALL)
    Q_q_matches = re.findall(Q_q_pattern, log_content, re.DOTALL)
    
    extracted_data = {
        'quad_s0': [],
        'Q_q': []
    }
    
    # 解析quad_s0数据
    for match in quad_s0_matches[:10]:  # 只取前10个样本
        try:
            # 清理数据字符串
            data_str = match.strip().replace('\n', '').replace(' ', '')
            # 分割数据
            values = [float(x) for x in data_str.split(',') if x.strip()]
            if len(values) == 10:  # 确保状态维度正确
                extracted_data['quad_s0'].append(values)
        except:
            continue
    
    # 解析Q_q数据
    for match in Q_q_matches[:10]:  # 只取前10个样本
        try:
            # 清理数据字符串
            data_str = match.strip().replace('\n', '').replace(' ', '')
            # 分割数据
            values = [float(x) for x in data_str.split(',') if x.strip()]
            if len(values) == 14:  # 确保代价权重维度正确 (10+4)
                extracted_data['Q_q'].append(values)
        except:
            continue
    
    return extracted_data

def update_test_script_with_real_data(extracted_data):
    """用提取的真实数据更新测试脚本"""
    
    if not extracted_data['quad_s0'] or not extracted_data['Q_q']:
        print("❌ 没有找到有效的数据，请检查日志格式")
        return False
    
    print(f"✅ 提取到 {len(extracted_data['quad_s0'])} 个quad_s0样本")
    print(f"✅ 提取到 {len(extracted_data['Q_q'])} 个Q_q样本")
    
    # 生成真实数据的Python代码
    quad_s0_code = "real_states = [\n"
    for i, state in enumerate(extracted_data['quad_s0'][:4]):  # 只用前4个
        quad_s0_code += f"            {state},\n"
    quad_s0_code += "        ]"
    
    Q_q_code = "real_Q_q = [\n"
    for i, weights in enumerate(extracted_data['Q_q'][:4]):  # 只用前4个
        Q_q_code += f"            {weights},\n"
    Q_q_code += "        ]"
    
    print("\n生成的真实数据代码:")
    print("=" * 60)
    print("quad_s0数据:")
    print(quad_s0_code)
    print("\nQ_q数据:")
    print(Q_q_code)
    print("=" * 60)
    
    # 读取当前测试脚本
    test_script_path = "/home/zhaoguodong/work/code/MAPPO-MPC-OmniDrones/OmniDrones/test_mpc_simple.py"
    
    try:
        with open(test_script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # 替换quad_s0数据
        old_real_states_pattern = r'real_states = \[.*?\]'
        new_real_states = quad_s0_code.strip()
        script_content = re.sub(old_real_states_pattern, new_real_states, script_content, flags=re.DOTALL)
        
        # 替换Q_q数据
        old_real_Q_q_pattern = r'real_Q_q = \[.*?\]'
        new_real_Q_q = Q_q_code.strip()
        script_content = re.sub(old_real_Q_q_pattern, new_real_Q_q, script_content, flags=re.DOTALL)
        
        # 写回文件
        with open(test_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"✅ 已更新测试脚本: {test_script_path}")
        return True
        
    except Exception as e:
        print(f"❌ 更新测试脚本失败: {e}")
        return False

def main():
    """主函数"""
    print("MPC训练数据提取工具")
    print("=" * 50)
    
    # 检查是否提供了日志文件路径
    if len(sys.argv) > 1:
        log_file_path = sys.argv[1]
        if os.path.exists(log_file_path):
            print(f"📖 从文件读取日志: {log_file_path}")
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
        else:
            print(f"❌ 日志文件不存在: {log_file_path}")
            return
    else:
        print("📝 请粘贴包含quad_s0和Q_q数据的日志内容")
        print("输入完成后，请输入 'END' 并按回车:")
        print("-" * 50)
        
        log_lines = []
        while True:
            try:
                line = input()
                if line.strip().upper() == 'END':
                    break
                log_lines.append(line)
            except KeyboardInterrupt:
                print("\n用户中断操作")
                return
            except EOFError:
                break
        
        log_content = '\n'.join(log_lines)
    
    if not log_content.strip():
        print("❌ 没有输入任何日志内容")
        return
    
    # 提取数据
    print("\n🔍 正在分析日志内容...")
    extracted_data = extract_data_from_logs(log_content)
    
    # 更新测试脚本
    if update_test_script_with_real_data(extracted_data):
        print("\n🎯 现在可以运行 test_mpc_simple.py 来重现问题!")
        print("运行命令: python test_mpc_simple.py")
    else:
        print("\n❌ 数据提取失败，请检查日志格式")

if __name__ == "__main__":
    main()
