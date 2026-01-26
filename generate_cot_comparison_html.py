#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CoT 修订前后对比展示 HTML 生成脚本

功能：
1. 从 CoT 蒸馏结果 JSON 文件中加载数据
2. 生成 HTML 展示页面，对比修订前后的 CoT
3. 支持人工打分和评测

展示内容：
- 图片
- Question (问题)
- Ground Truth Answer (标准答案)
- 修订前 CoT + 回答
- 修订后 CoT + 回答
- 模型的修改原因
- 模型的质量评分
- 人工打分区域
"""

import json
import os
import random
import shutil
from datetime import datetime
from string import Template
from pathlib import Path


# ==================== 配置区域 ====================
# 输入JSON文件路径
INPUT_JSON_PATH = "/home/maxzhang/datapipeline/temp_images/cot_distilled_output.json"

# 输出目录
OUTPUT_DIR = "/home/maxzhang/datapipeline/temp_images"

# 随机抽取数量 (设为 None 则使用全部数据)
SAMPLE_COUNT = 50

# 随机种子
RANDOM_SEED = 42
# ==================== 配置区域结束 ====================


def load_json_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def escape_html(text):
    """转义HTML特殊字符"""
    if text is None:
        return ""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
            .replace("\n", "<br>"))


def format_cot_display(cot_text):
    """格式化 CoT 显示，将 </think> 分隔符高亮"""
    if not cot_text:
        return ""
    
    escaped = escape_html(cot_text)
    # 将 </think> 替换为高亮的分隔符
    escaped = escaped.replace("&lt;/think&gt;", 
        '<span class="think-separator">💭 思考结束 → 开始回答</span>')
    return escaped


def generate_html(data_list, output_path):
    """生成HTML展示页面"""
    
    html_template = Template('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CoT 修订对比展示 - 修订前 vs 修订后</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .info-bar {
            background: rgba(255,255,255,0.95);
            padding: 15px 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .info-bar span {
            color: #555;
            font-size: 14px;
        }
        
        .save-section {
            background: rgba(255,255,255,0.95);
            padding: 20px 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .save-btn {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .save-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
        }
        
        .progress-info {
            color: #0f3460;
            font-weight: bold;
        }
        
        /* 统计区域 */
        .stats-section {
            background: linear-gradient(135deg, #e94560, #0f3460);
            padding: 25px;
            border-radius: 16px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            color: white;
        }
        
        .stats-section h2 {
            margin-bottom: 20px;
            font-size: 1.5em;
            text-align: center;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.15);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        
        .stat-card .stat-label {
            font-size: 13px;
            opacity: 0.9;
            margin-bottom: 8px;
        }
        
        .stat-card .stat-value {
            font-size: 2em;
            font-weight: bold;
        }
        
        .stat-card.before .stat-value {
            color: #ffd93d;
        }
        
        .stat-card.after .stat-value {
            color: #6bcb77;
        }
        
        .data-row {
            background: rgba(255,255,255,0.98);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .data-row:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.25);
        }
        
        .data-row.evaluated {
            border: 3px solid #28a745;
        }
        
        .data-row.revised {
            border-left: 5px solid #e94560;
        }
        
        .row-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .row-number {
            background: linear-gradient(135deg, #0f3460, #e94560);
            color: white;
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 16px;
        }
        
        .model-score {
            background: linear-gradient(135deg, #ffd93d, #ff6b6b);
            color: #333;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
        
        .revised-badge {
            background: #e94560;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        
        .eval-status {
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        
        .eval-status.pending {
            background: #ffc107;
            color: #333;
        }
        
        .eval-status.done {
            background: #28a745;
            color: white;
        }
        
        .image-section {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .image-section h3 {
            color: #444;
            margin-bottom: 10px;
            font-size: 14px;
            background: #f5f5f5;
            padding: 8px;
            border-radius: 6px;
            display: inline-block;
        }
        
        .image-section img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        
        .image-section img:hover {
            transform: scale(1.02);
        }
        
        .content-section {
            display: grid;
            grid-template-columns: 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .content-block {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px 20px;
            border-left: 4px solid #0f3460;
        }
        
        .content-block h4 {
            color: #0f3460;
            margin-bottom: 10px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .content-block .content-text {
            color: #333;
            line-height: 1.8;
            font-size: 14px;
            max-height: 200px;
            overflow-y: auto;
            padding-right: 10px;
        }
        
        /* CoT 对比区域 */
        .cot-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .cot-block {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            position: relative;
        }
        
        .cot-block.before {
            border: 2px solid #ffd93d;
            background: linear-gradient(135deg, #fffdf0 0%, #fff9e6 100%);
        }
        
        .cot-block.after {
            border: 2px solid #6bcb77;
            background: linear-gradient(135deg, #f0fff4 0%, #e6ffed 100%);
        }
        
        .cot-block h4 {
            margin-bottom: 15px;
            font-size: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }
        
        .cot-block.before h4 {
            color: #b8860b;
        }
        
        .cot-block.after h4 {
            color: #228b22;
        }
        
        .cot-block .content-text {
            color: #333;
            line-height: 1.9;
            font-size: 14px;
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
        }
        
        .think-separator {
            display: block;
            background: linear-gradient(90deg, #667eea, #764ba2);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            margin: 15px 0;
            font-weight: bold;
            font-size: 13px;
            text-align: center;
        }
        
        /* 模型修改原因 */
        .revision-info {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
            border: 2px solid #ffc107;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .revision-info h4 {
            color: #856404;
            margin-bottom: 15px;
            font-size: 15px;
        }
        
        .revision-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .revision-item {
            background: white;
            padding: 12px 15px;
            border-radius: 8px;
        }
        
        .revision-item label {
            display: block;
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        
        .revision-item .value {
            font-size: 14px;
            color: #333;
            line-height: 1.6;
        }
        
        /* 评测区域样式 */
        .evaluation-section {
            margin-top: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-radius: 12px;
            border: 2px dashed #2196f3;
        }
        
        .evaluation-section h4 {
            color: #1565c0;
            margin-bottom: 15px;
            font-size: 16px;
        }
        
        .score-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 15px;
        }
        
        .score-block {
            background: white;
            padding: 15px 20px;
            border-radius: 12px;
            text-align: center;
        }
        
        .score-block.before {
            border: 2px solid #ffd93d;
        }
        
        .score-block.after {
            border: 2px solid #6bcb77;
        }
        
        .score-block label {
            display: block;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 14px;
        }
        
        .score-block.before label {
            color: #b8860b;
        }
        
        .score-block.after label {
            color: #228b22;
        }
        
        .score-select {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: pointer;
            background: white;
            transition: all 0.2s;
        }
        
        .score-select:focus {
            outline: none;
            border-color: #2196f3;
        }
        
        .reason-input {
            width: 100%;
            min-height: 80px;
            padding: 12px 15px;
            border: 2px solid #2196f3;
            border-radius: 10px;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.2s;
        }
        
        .reason-input:focus {
            outline: none;
            border-color: #0d47a1;
        }
        
        .content-text::-webkit-scrollbar {
            width: 6px;
        }
        
        .content-text::-webkit-scrollbar-track {
            background: #e0e0e0;
            border-radius: 3px;
        }
        
        .content-text::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 3px;
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            cursor: pointer;
        }
        
        .modal img {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 95%;
            max-height: 95%;
            border-radius: 8px;
        }
        
        .modal-close {
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            cursor: pointer;
        }
        
        .toast {
            position: fixed;
            bottom: 30px;
            right: 30px;
            padding: 15px 25px;
            background: #28a745;
            color: white;
            border-radius: 10px;
            font-weight: bold;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 2000;
            display: none;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @media (max-width: 1200px) {
            .cot-comparison {
                grid-template-columns: 1fr;
            }
            .score-grid {
                grid-template-columns: 1fr;
            }
            .revision-grid {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            h1 {
                font-size: 1.8em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 CoT 修订对比展示</h1>
        
        <div class="info-bar">
            <span><strong>数据总数:</strong> $total_count 条</span>
            <span><strong>被修订数据:</strong> $revised_count 条</span>
            <span><strong>生成时间:</strong> $generation_time</span>
            <span><strong>说明:</strong> 🟡黄色=修订前, 🟢绿色=修订后，请分别打分(1-10分)</span>
        </div>
        
        <div class="save-section">
            <span class="progress-info" id="progressInfo">已完成: 0 / $total_count</span>
            <button class="save-btn" onclick="saveEvaluation()">💾 保存评测结果</button>
        </div>
        
        <!-- 统计区域 -->
        <div class="stats-section" id="statsSection">
            <h2>📈 评分统计</h2>
            <div class="stats-grid">
                <div class="stat-card before">
                    <div class="stat-label">🟡 修订前 CoT 均分</div>
                    <div class="stat-value" id="beforeAvg">--</div>
                </div>
                <div class="stat-card after">
                    <div class="stat-label">🟢 修订后 CoT 均分</div>
                    <div class="stat-value" id="afterAvg">--</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">📊 已评分数量</div>
                    <div class="stat-value" id="scoredCount">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">🎯 提升分数</div>
                    <div class="stat-value" id="scoreDiff">--</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">🤖 模型平均评分</div>
                    <div class="stat-value" id="modelAvg">$model_avg_score</div>
                </div>
            </div>
        </div>
        
        $data_rows
    </div>
    
    <!-- 图片放大模态框 -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <img id="modalImage" src="" alt="放大图片">
    </div>
    
    <!-- 提示框 -->
    <div id="toast" class="toast"></div>
    
    <script>
        // 原始数据存储
        const originalData = $original_data;
        
        // 更新进度和统计
        function updateProgress() {
            let completed = 0;
            let beforeTotal = 0;
            let afterTotal = 0;
            let scoredCount = 0;
            const total = originalData.length;
            
            originalData.forEach((item, index) => {
                const beforeScore = document.getElementById('before_score_' + index);
                const afterScore = document.getElementById('after_score_' + index);
                
                const bScore = beforeScore ? parseInt(beforeScore.value) : 0;
                const aScore = afterScore ? parseInt(afterScore.value) : 0;
                
                if (bScore > 0 && aScore > 0) {
                    completed++;
                    beforeTotal += bScore;
                    afterTotal += aScore;
                    scoredCount++;
                    document.getElementById('row_' + index).classList.add('evaluated');
                    document.getElementById('status_' + index).className = 'eval-status done';
                    document.getElementById('status_' + index).textContent = '✓ 已评分';
                } else {
                    document.getElementById('row_' + index).classList.remove('evaluated');
                    document.getElementById('status_' + index).className = 'eval-status pending';
                    document.getElementById('status_' + index).textContent = '待评分';
                }
            });
            
            document.getElementById('progressInfo').textContent = '已完成: ' + completed + ' / ' + total;
            document.getElementById('scoredCount').textContent = scoredCount;
            
            // 计算均分
            if (scoredCount > 0) {
                const beforeAvg = (beforeTotal / scoredCount).toFixed(2);
                const afterAvg = (afterTotal / scoredCount).toFixed(2);
                const diff = (afterAvg - beforeAvg).toFixed(2);
                
                document.getElementById('beforeAvg').textContent = beforeAvg;
                document.getElementById('afterAvg').textContent = afterAvg;
                document.getElementById('scoreDiff').textContent = (diff > 0 ? '+' : '') + diff;
                
                // 根据分差设置颜色
                const diffElement = document.getElementById('scoreDiff');
                if (diff > 0) {
                    diffElement.style.color = '#6bcb77';
                } else if (diff < 0) {
                    diffElement.style.color = '#ff6b6b';
                } else {
                    diffElement.style.color = 'white';
                }
            } else {
                document.getElementById('beforeAvg').textContent = '--';
                document.getElementById('afterAvg').textContent = '--';
                document.getElementById('scoreDiff').textContent = '--';
            }
        }
        
        // 显示提示
        function showToast(message, isError = false) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.style.background = isError ? '#dc3545' : '#28a745';
            toast.style.display = 'block';
            setTimeout(() => {
                toast.style.display = 'none';
            }, 3000);
        }
        
        // 保存评测结果
        function saveEvaluation() {
            let beforeTotal = 0;
            let afterTotal = 0;
            let scoredCount = 0;
            
            const results = {
                evaluation_time: new Date().toLocaleString('zh-CN'),
                total_count: originalData.length,
                completed_count: 0,
                statistics: {
                    before_average: 0,
                    after_average: 0,
                    improvement: 0
                },
                evaluations: []
            };
            
            originalData.forEach((item, index) => {
                const beforeScoreEl = document.getElementById('before_score_' + index);
                const afterScoreEl = document.getElementById('after_score_' + index);
                const reasonTextarea = document.getElementById('reason_' + index);
                
                const beforeScore = beforeScoreEl ? parseInt(beforeScoreEl.value) || 0 : 0;
                const afterScore = afterScoreEl ? parseInt(afterScoreEl.value) || 0 : 0;
                const reason = reasonTextarea ? reasonTextarea.value.trim() : '';
                
                if (beforeScore > 0 && afterScore > 0) {
                    results.completed_count++;
                    beforeTotal += beforeScore;
                    afterTotal += afterScore;
                    scoredCount++;
                }
                
                results.evaluations.push({
                    index: index + 1,
                    id: item.id,
                    question: item.question,
                    ground_truth_answer: item.answer,
                    model_quality_score: item.model_score,
                    was_revised: item.was_revised,
                    human_evaluation: {
                        before_score: beforeScore,
                        after_score: afterScore,
                        reason: reason
                    }
                });
            });
            
            // 计算统计数据
            if (scoredCount > 0) {
                results.statistics.before_average = parseFloat((beforeTotal / scoredCount).toFixed(2));
                results.statistics.after_average = parseFloat((afterTotal / scoredCount).toFixed(2));
                results.statistics.improvement = parseFloat((results.statistics.after_average - results.statistics.before_average).toFixed(2));
            }
            
            // 生成JSON文件并下载
            const jsonContent = JSON.stringify(results, null, 2);
            const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            const filename = 'cot_evaluation_result_' + timestamp + '.json';
            
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
            
            showToast('✅ 评测结果已保存: ' + filename);
        }
        
        // 图片放大
        function openModal(imgSrc) {
            document.getElementById('modalImage').src = imgSrc;
            document.getElementById('imageModal').style.display = 'block';
        }
        
        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
        
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeModal();
            }
        });
        
        // 监听打分变化
        document.addEventListener('change', function(e) {
            if (e.target.classList.contains('score-select')) {
                updateProgress();
            }
        });
        
        // 初始化进度
        updateProgress();
    </script>
</body>
</html>''')

    row_template = Template('''
        <div class="data-row $revised_class" id="row_$index">
            <div class="row-header">
                <span class="row-number">📋 数据 #$row_num</span>
                <span class="model-score">🤖 模型评分: $model_score</span>
                $revised_badge
                <span class="eval-status pending" id="status_$index">待评分</span>
            </div>
            
            <!-- 图片区域 -->
            <div class="image-section">
                <h3>🖼️ 图片</h3><br>
                $image_html
            </div>
            
            <!-- 问题和标准答案 -->
            <div class="content-section">
                <div class="content-block">
                    <h4>❓ 问题 (Question)</h4>
                    <div class="content-text">$question</div>
                </div>
                
                <div class="content-block">
                    <h4>✅ 标准答案 (Ground Truth)</h4>
                    <div class="content-text">$answer</div>
                </div>
            </div>
            
            <!-- CoT 对比区域 -->
            <div class="cot-comparison">
                <div class="cot-block before">
                    <h4>🟡 修订前 CoT + 回答</h4>
                    <div class="content-text">$cot_before</div>
                </div>
                <div class="cot-block after">
                    <h4>🟢 修订后 CoT + 回答</h4>
                    <div class="content-text">$cot_after</div>
                </div>
            </div>
            
            <!-- 模型修改信息 -->
            <div class="revision-info">
                <h4>🔧 模型修订信息</h4>
                <div class="revision-grid">
                    <div class="revision-item">
                        <label>是否被修订</label>
                        <div class="value">$was_revised</div>
                    </div>
                    <div class="revision-item">
                        <label>模型质量评分</label>
                        <div class="value">$model_score / 1.0</div>
                    </div>
                    <div class="revision-item">
                        <label>修订原因</label>
                        <div class="value">$revision_reason</div>
                    </div>
                    <div class="revision-item">
                        <label>检测到的幻觉</label>
                        <div class="value">$hallucination</div>
                    </div>
                </div>
            </div>
            
            <!-- 人工评测区域 -->
            <div class="evaluation-section">
                <h4>📋 人工评测区域</h4>
                
                <div class="score-grid">
                    <div class="score-block before">
                        <label>🟡 修订前 CoT 评分 (1-10分)</label>
                        <select class="score-select" id="before_score_$index">
                            <option value="">请选择分数</option>
                            <option value="1">1分 - 非常差</option>
                            <option value="2">2分</option>
                            <option value="3">3分</option>
                            <option value="4">4分</option>
                            <option value="5">5分 - 一般</option>
                            <option value="6">6分</option>
                            <option value="7">7分</option>
                            <option value="8">8分</option>
                            <option value="9">9分</option>
                            <option value="10">10分 - 完美</option>
                        </select>
                    </div>
                    <div class="score-block after">
                        <label>🟢 修订后 CoT 评分 (1-10分)</label>
                        <select class="score-select" id="after_score_$index">
                            <option value="">请选择分数</option>
                            <option value="1">1分 - 非常差</option>
                            <option value="2">2分</option>
                            <option value="3">3分</option>
                            <option value="4">4分</option>
                            <option value="5">5分 - 一般</option>
                            <option value="6">6分</option>
                            <option value="7">7分</option>
                            <option value="8">8分</option>
                            <option value="9">9分</option>
                            <option value="10">10分 - 完美</option>
                        </select>
                    </div>
                </div>
                
                <div>
                    <label style="font-weight: bold; color: #1565c0; display: block; margin-bottom: 10px;">评测理由（可选）：</label>
                    <textarea class="reason-input" id="reason_$index" placeholder="请在此输入您的评测理由，例如：思考过程是否自然、是否有幻觉、修订是否有效等..."></textarea>
                </div>
            </div>
        </div>''')

    # 收集原始数据用于JavaScript
    original_data_for_js = []
    revised_count = 0
    model_scores = []
    
    data_rows = []
    for i, item in enumerate(data_list):
        # 获取图片路径
        image_path = item.get('image_path', '')
        if image_path:
            image_name = os.path.basename(image_path)
            image_html = f'<img src="{image_name}" alt="图片" onclick="openModal(this.src)">'
        else:
            image_html = '<p style="color: #999;">无图片</p>'
        
        # 获取数据
        question = item.get('question', item.get('refined_question', ''))
        answer = item.get('answer', item.get('refined_answer', ''))
        cot_before = item.get('cot_combined', '')
        cot_after = item.get('final_cot_combined', cot_before)
        was_revised = item.get('was_revised', False)
        revision_reason = item.get('revision_reason', '无')
        hallucination = item.get('hallucination_detected', '无')
        model_score = item.get('quality_score', 0)
        
        if was_revised:
            revised_count += 1
        
        if model_score:
            model_scores.append(float(model_score))
        
        # 为JavaScript存储原始数据
        original_data_for_js.append({
            'id': item.get('id', i + 1),
            'question': question,
            'answer': answer,
            'model_score': model_score,
            'was_revised': was_revised
        })
        
        row_html = row_template.substitute(
            index=i,
            row_num=i + 1,
            revised_class='revised' if was_revised else '',
            revised_badge='<span class="revised-badge">🔄 已修订</span>' if was_revised else '',
            image_html=image_html,
            question=escape_html(question),
            answer=escape_html(answer),
            cot_before=format_cot_display(cot_before),
            cot_after=format_cot_display(cot_after),
            was_revised='✅ 是' if was_revised else '❌ 否',
            model_score=f"{model_score:.2f}" if model_score else "N/A",
            revision_reason=escape_html(revision_reason) if revision_reason else '无',
            hallucination=escape_html(hallucination) if hallucination else '无'
        )
        data_rows.append(row_html)
    
    # 计算模型平均评分
    model_avg = f"{sum(model_scores) / len(model_scores):.2f}" if model_scores else "--"
    
    # 将原始数据转为JSON字符串
    original_data_json = json.dumps(original_data_for_js, ensure_ascii=False)
    
    html_content = html_template.substitute(
        total_count=len(data_list),
        revised_count=revised_count,
        generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        model_avg_score=model_avg,
        data_rows='\n'.join(data_rows),
        original_data=original_data_json
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML页面已生成: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("🔬 CoT 修订对比 HTML 生成器")
    print("=" * 60)
    
    # 设置随机种子
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    
    # 加载数据
    print(f"\n📂 加载数据: {INPUT_JSON_PATH}")
    data = load_json_data(INPUT_JSON_PATH)
    print(f"   总数据量: {len(data)} 条")
    
    # 过滤有效数据
    valid_data = [item for item in data if item.get('step1_status') == 'success']
    print(f"   有效数据: {len(valid_data)} 条")
    
    # 随机抽样
    if SAMPLE_COUNT and SAMPLE_COUNT < len(valid_data):
        sampled_data = random.sample(valid_data, SAMPLE_COUNT)
        print(f"   随机抽取: {SAMPLE_COUNT} 条")
    else:
        sampled_data = valid_data
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 生成HTML
    html_path = os.path.join(OUTPUT_DIR, "cot_comparison.html")
    generate_html(sampled_data, html_path)
    
    print(f"\n🎉 完成！请在浏览器中打开: {html_path}")


if __name__ == "__main__":
    main()
