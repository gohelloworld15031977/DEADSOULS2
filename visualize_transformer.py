"""
Визуализация архитектуры трансформера GPT-2 с механизмами обучения
DeadSouls Project - Gogol Fine-tuning
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, ConnectionPatch
import numpy as np

# Установить кодировку для matplotlib
plt.rcParams['axes.unicode_minus'] = False

def draw_transformer_architecture():
    """Визуализация полной архитектуры трансформера"""
    
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle('Архитектура трансформера GPT-2 + Механизмы обучения', 
                 fontsize=20, fontweight='bold')
    
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Цветовая схема
    colors = {
        'input': '#FF6B6B',
        'embedding': '#4ECDC4',
        'attention': '#95E1D3',
        'ffn': '#FCE38A',
        'normalization': '#F38181',
        'output': '#AA96DA',
        'lora': '#FFA07A',
        'gradient': '#2EC4B6',
        'backprop': '#E71D36',
        'optimization': '#011627'
    }
    
    # ==================== ВХОДНЫЕ ДАННЫЕ ====================
    draw_box(ax, 40, 92, 20, 4, 'Токены\n(идиот, приехал,\nЧичиков...)', 
             colors['input'], 'Вход')
    
    # ==================== EMBEDDING LAYER ====================
    draw_box(ax, 40, 84, 20, 4, 'Token Embeddings\n(768 измерений)', 
             colors['embedding'], 'Эмбеддинги')
    
    draw_box(ax, 65, 84, 20, 4, 'Positional Embeddings\n(позиционные коды)', 
             colors['embedding'], 'Позиции')
    
    # ==================== GPT-2 BLOCKS (12 слоёв) ====================
    y_start = 78
    block_height = 5
    num_layers = 12
    
    for i in range(num_layers):
        y = y_start - i * 3.5
        
        # Layer Normalization 1
        draw_box(ax, 35, y + 3, 30, 2, f'LayerNorm 1\n(ε=1e-12)', 
                 colors['normalization'], f'Слой {i+1}')
        
        # Multi-Head Attention
        draw_box(ax, 35, y, 30, 3, 
                 'Multi-Head Self-Attention\n'
                 '12 голов × 64 измерения\n'
                 'Q, K, V проекции\n'
                 'Attention(Q,K,V) = softmax(QKᵀ/√d)V', 
                 colors['attention'], f'Внимание {i+1}')
        
        # Layer Normalization 2
        draw_box(ax, 35, y - 3.5, 30, 2, f'LayerNorm 2\n(ε=1e-12)', 
                 colors['normalization'], f'Слой {i+1}')
        
        # Feed-Forward Network
        draw_box(ax, 35, y - 6.5, 30, 3, 
                 'Feed-Forward Network\n'
                 'Linear → GELU → Linear\n'
                 '768 → 3072 → 768', 
                 colors['ffn'], f'FFN {i+1}')
        
        # LoRA адаптеры (показать только в некоторых слоях)
        if i in [0, 3, 6, 9, 11]:
            draw_box(ax, 70, y, 12, 3, 
                     'LoRA\n'
                     'r=8, α=16\n'
                     '∆W = BA\n'
                     'c_attn, c_proj, c_fc', 
                     colors['lora'], f'LoRA {i+1}', linewidth=2)
    
    # ==================== FINAL LAYERS ====================
    y_final = y_start - num_layers * 3.5 - 3
    
    draw_box(ax, 40, y_final, 20, 3, 
             'LayerNorm\n(финальный)', 
             colors['normalization'], 'Финал')
    
    draw_box(ax, 40, y_final - 4, 20, 3, 
             'Linear Head\n768 → 32000\n(словарь)', 
             colors['output'], 'Логиты')
    
    draw_box(ax, 40, y_final - 8, 20, 3, 
             'Softmax\nP(token|context)', 
             colors['output'], 'Вероятности')
    
    # ==================== LOSS CALCULATION ====================
    draw_box(ax, 65, y_final - 8, 20, 3, 
             'CrossEntropyLoss\n'
             'L = -log(P(true_token))', 
             colors['optimization'], 'Loss')
    
    # ==================== BACKPROPAGATION ====================
    draw_box(ax, 10, y_final - 8, 20, 3, 
             'BACKPROPAGATION\n'
             '∇L(θ) вычисляется\n'
             'через все слои', 
             colors['backprop'], 'Backward', 
             linestyle='--', linewidth=2)
    
    # ==================== GRADIENT DESCENT ====================
    draw_box(ax, 10, y_final - 12, 20, 3, 
             'GRADIENT DESCENT\n'
             'θ = θ - lr·∇L(θ)\n'
             'AdamW optimizer\n'
             'lr=1e-4, weight_decay=0.01', 
             colors['optimization'], 'Optimization')
    
    # ==================== ARROWS ====================
    # Forward pass
    for i in range(num_layers):
        y = y_start - i * 3.5
        # Arrow from previous layer to current
        if i == 0:
            ax.annotate('', xy=(50, y + 3), xytext=(50, y + 10),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
        else:
            prev_y = y_start - (i-1) * 3.5
            ax.annotate('', xy=(50, y + 3), xytext=(50, prev_y - 6.5),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # Arrow inside attention block
        ax.annotate('', xy=(50, y), xytext=(50, y + 3),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # Arrow to FFN
        ax.annotate('', xy=(50, y - 3.5), xytext=(50, y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # Arrow after FFN
        if i < num_layers - 1:
            ax.annotate('', xy=(50, y - 6.5), xytext=(50, y - 3.5),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Final arrows
    final_y = y_start - num_layers * 3.5
    ax.annotate('', xy=(50, y_final), xytext=(50, final_y - 3),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(50, y_final - 4), xytext=(50, y_final),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(50, y_final - 8), xytext=(50, y_final - 4),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Loss arrow
    ax.annotate('', xy=(65, y_final - 8), xytext=(60, y_final - 8),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Backprop arrow (обратный путь)
    for i in range(num_layers):
        y = y_start - i * 3.5
        if i == num_layers - 1:
            ax.annotate('', xy=(20, y_final - 8), xytext=(20, final_y - 3),
                       arrowprops=dict(arrowstyle='->', color=colors['backprop'], lw=2, 
                                      linestyle='--', mutation_scale=20))
        prev_y = y_start - (i-1) * 3.5 if i > 0 else y_final
        ax.annotate('', xy=(20, y - 6.5), xytext=(20, prev_y - 3),
                   arrowprops=dict(arrowstyle='->', color=colors['backprop'], lw=2,
                                  linestyle='--', mutation_scale=20))
    
    # Gradient descent arrow
    ax.annotate('', xy=(20, y_final - 12), xytext=(20, y_final - 8),
               arrowprops=dict(arrowstyle='->', color=colors['optimization'], lw=2))
    
    # ==================== LEGEND ====================
    legend_data = [
        ('Входные данные', colors['input']),
        ('Эмбеддинги', colors['embedding']),
        ('Multi-Head Attention', colors['attention']),
        ('Feed-Forward Network', colors['ffn']),
        ('Layer Normalization', colors['normalization']),
        ('Output Head', colors['output']),
        ('LoRA адаптеры', colors['lora']),
        ('Backpropagation', colors['backprop']),
        ('Градиентный спуск', colors['optimization'])
    ]
    
    for i, (label, color) in enumerate(legend_data):
        rect = patches.Rectangle((5, 85 - i*2), 2, 1, facecolor=color, 
                                edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(8, 85.5 - i*2, label, fontsize=9, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('transformer_architecture.png', dpi=150, bbox_inches='tight')
    print("[OK] Сохранено: transformer_architecture.png")
    plt.show()


def draw_attention_mechanism():
    """Детальная визуализация механизма внимания"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Механизм Multi-Head Self-Attention', fontsize=16, fontweight='bold')
    
    # === Q, K, V проекции ===
    ax1 = axes[0]
    ax1.set_title('1. Q, K, V Проекции')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Input embeddings
    for i in range(5):
        rect = patches.Rectangle((1, 8-i*1.5), 2, 1, facecolor='#4ECDC4', 
                                edgecolor='black')
        ax1.add_patch(rect)
        ax1.text(2, 8.5-i*1.5, f'emb{i}', ha='center', va='center', fontsize=8)
    
    # Q, K, V matrices
    for i, (label, color) in enumerate([('Q', '#FF6B6B'), ('K', '#4ECDC4'), ('V', '#95E1D3')]):
        for j in range(3):
            rect = patches.Rectangle((5+j, 2+i*2.5), 1, 2, facecolor=color, 
                                    edgecolor='black', alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(5.5+j, 3+i*2.5, label, ha='center', va='center', fontsize=8, 
                    fontweight='bold')
    
    ax1.text(2, 0, 'Input (5 токенов × 768)', ha='center', fontsize=10)
    ax1.text(7, 0, 'W_Q, W_K, W_V (768×64)', ha='center', fontsize=10)
    
    # === Attention Scores ===
    ax2 = axes[1]
    ax2.set_title('2. Attention Scores')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Attention matrix
    matrix = np.array([
        [0.3, 0.1, 0.05, 0.05, 0.1],
        [0.2, 0.3, 0.1, 0.05, 0.1],
        [0.1, 0.2, 0.3, 0.1, 0.05],
        [0.05, 0.1, 0.2, 0.3, 0.1],
        [0.1, 0.05, 0.1, 0.2, 0.3]
    ])
    
    im = ax2.imshow(matrix, cmap='YlOrRd', extent=[0, 5, 0, 5])
    for i in range(5):
        for j in range(5):
            ax2.text(j+0.5, i+0.5, f'{matrix[i,j]:.2f}', ha='center', va='center', 
                    fontsize=8, fontweight='bold')
    
    ax2.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax2.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax2.set_xticklabels(['T0', 'T1', 'T2', 'T3', 'T4'], fontsize=8)
    ax2.set_yticklabels(['T0', 'T1', 'T2', 'T3', 'T4'], fontsize=8)
    ax2.set_xlabel('Keys', fontsize=10)
    ax2.set_ylabel('Queries', fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=8)
    
    # === Weighted Sum ===
    ax3 = axes[2]
    ax3.set_title('3. Weighted Sum → Output')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    # Attention weights
    for i in range(5):
        rect = patches.Rectangle((0.5, 8-i*1.5), 2, 1, facecolor='#95E1D3', 
                                edgecolor='black', alpha=0.5)
        ax3.add_patch(rect)
        ax3.text(1.5, 8.5-i*1.5, f'a{i}', ha='center', va='center', fontsize=8)
    
    # Value vectors
    for i in range(5):
        rect = patches.Rectangle((4, 8-i*1.5), 2, 1, facecolor='#4ECDC4', 
                                edgecolor='black', alpha=0.7)
        ax3.add_patch(rect)
        ax3.text(5, 8.5-i*1.5, f'v{i}', ha='center', va='center', fontsize=8)
    
    # Output
    rect = patches.Rectangle((7.5, 6), 2, 3, facecolor='#FF6B6B', 
                            edgecolor='black', linewidth=2)
    ax3.add_patch(rect)
    ax3.text(8.5, 7.5, 'Output\n(768)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Arrows
    for i in range(5):
        ax3.annotate('', xy=(4, 8.5-i*1.5), xytext=(2.5, 8.5-i*1.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    for i in range(5):
        ax3.annotate('', xy=(7.5, 7.5), xytext=(6, 8.5-i*1.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    plt.tight_layout()
    plt.savefig('attention_mechanism.png', dpi=150, bbox_inches='tight')
    print("[OK] Сохранено: attention_mechanism.png")
    plt.show()


def draw_training_loop():
    """Визуализация цикла обучения с градиентным спуском"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle('Цикл обучения: Forward → Backward → Optimization', 
                 fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # === FORWARD PASS ===
    draw_box(ax, 20, 85, 20, 5, 'FORWARD PASS\n'
             '1. Батч токенов\n'
             '2. Модель → предсказания\n'
             '3. Loss = CrossEntropy', 
             '#4ECDC4', 'Forward')
    
    # === BACKPROPAGATION ===
    draw_box(ax, 50, 85, 20, 5, 'BACKPROPAGATION\n'
             '1. loss.backward()\n'
             '2. ∇L(θ) для всех θ\n'
             '3. Градиенты в .grad', 
             '#E71D36', 'Backward', linestyle='--')
    
    # === GRADIENT ACCUMULATION ===
    draw_box(ax, 80, 85, 15, 5, 'GRADIENT\nACCUMULATION\n'
             'Суммирование за 4 шага\n'
             'Эффективный batch=4', 
             '#FFA07A', 'Accum')
    
    # === GRADIENT CLIPPING ===
    draw_box(ax, 20, 65, 20, 5, 'GRADIENT CLIPPING\n'
             'if grad_norm > 0.5:\n'
             '  grad *= 0.5/grad_norm\n'
             'Предотвращение взрыва', 
             '#95E1D3', 'Clip')
    
    # === OPTIMIZER STEP ===
    draw_box(ax, 50, 65, 20, 5, 'OPTIMIZER STEP\n'
             'AdamW: θ = θ - lr·∇L(θ)\n'
             'weight_decay = 0.01\n'
             'lr = 1e-4 (scheduler)', 
             '#2EC4B6', 'Optimize')
    
    # === ZERO GRADIENTS ===
    draw_box(ax, 80, 65, 15, 5, 'ZERO GRADIENTS\n'
             'optimizer.zero_grad()\n'
             'Очистка для нового шага', 
             '#FCE38A', 'Zero')
    
    # === ARROWS ===
    # Forward to Backward
    ax.annotate('', xy=(40, 85), xytext=(30, 90),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    # Backward to Accumulation
    ax.annotate('', xy=(60, 85), xytext=(72, 90),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Accumulation to Clip
    ax.annotate('', xy=(75, 85), xytext=(70, 70),
               arrowprops=dict(arrowstyle='->', color='black', lw=2, 
                              linestyle='--', mutation_scale=20))
    # Clip to Optimize
    ax.annotate('', xy=(30, 70), xytext=(48, 70),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    # Optimize to Zero
    ax.annotate('', xy=(60, 70), xytext=(72, 70),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Zero to Forward (loop)
    ax.annotate('', xy=(80, 70), xytext=(85, 90),
               arrowprops=dict(arrowstyle='->', color='black', lw=2, 
                              mutation_scale=20))
    ax.annotate('', xy=(85, 90), xytext=(20, 90),
               arrowprops=dict(arrowstyle='->', color='black', lw=2, 
                              mutation_scale=20))
    ax.annotate('', xy=(20, 90), xytext=(20, 85),
               arrowprops=dict(arrowstyle='->', color='black', lw=2, 
                              mutation_scale=20))
    
    # === METRICS ===
    draw_box(ax, 20, 45, 20, 5, 'METRICS\n'
             'Loss: 4.69 → 3.75\n'
             'Grad Norm: 3.5 → 1.2\n'
             'Eval Loss: 3.84 → 3.75', 
             '#AA96FC', 'Metrics')
    
    # === EARLY STOPPING ===
    draw_box(ax, 50, 45, 20, 5, 'EARLY STOPPING\n'
             'patience=2\n'
             'threshold=0.01\n'
             'Мониторинг eval_loss', 
             '#FF6B9D', 'Early Stop')
    
    # === LOADING SAVING ===
    draw_box(ax, 80, 45, 15, 5, 'SAVE CHECKPOINT\n'
             'checkpoint-750/\n'
             'adapter_model.safetensors\n'
             'Каждую эпоху', 
             '#FFA07A', 'Save')
    
    plt.tight_layout()
    plt.savefig('training_loop.png', dpi=150, bbox_inches='tight')
    print("[OK] Сохранено: training_loop.png")
    plt.show()


def draw_lora_adapters():
    """Визуализация LoRA адаптеров"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('LoRA (Low-Rank Adaptation)', fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Original weight matrix
    draw_box(ax, 10, 70, 25, 20, 'Original Weight W\n'
             'Shape: (768, 768)\n'
             '589,824 параметров\n'
             '❌ ЗАМОРОЖЕНО', 
             '#FF6B6B', 'Original', linewidth=3)
    
    # LoRA A matrix
    draw_box(ax, 45, 80, 15, 8, 'LoRA B\n'
             'Shape: (768, 8)\n'
             '6,144 параметров', 
             '#4ECDC4', 'LoRA B')
    
    draw_box(ax, 45, 65, 15, 8, 'LoRA A\n'
             'Shape: (8, 768)\n'
             '6,144 параметров', 
             '#4ECDC4', 'LoRA A')
    
    # Result
    draw_box(ax, 75, 70, 15, 20, 'Output\n'
             'h = Wx + BAx\n'
             '12,288 обучаемых\n'
             'параметров (на слой)', 
             '#2EC4B6', 'Output')
    
    # Arrows
    ax.annotate('', xy=(60, 80), xytext=(75, 80),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(60, 65), xytext=(75, 65),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Input
    draw_box(ax, 10, 45, 20, 10, 'Input x\n'
             '(batch, seq, 768)', 
             '#95E1D3', 'Input')
    
    ax.annotate('', xy=(35, 50), xytext=(10, 70),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(35, 50), xytext=(45, 70),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Formula
    ax.text(50, 30, 'Формула LoRA:', fontsize=12, fontweight='bold')
    ax.text(50, 25, 'W' + '_new' + ' = W + ∆W = W + BA', fontsize=14, 
           fontfamily='serif')
    ax.text(50, 18, 'Где: r=8 (ранг), α=16 (масштаб)', fontsize=10)
    
    # Benefits
    benefits = [
        '✓ 98% меньше параметров для обучения',
        '✓ Быстрое переключение между задачами',
        '✓ Меньше памяти для оптимизатора',
        '✓ Сохранение базовой модели'
    ]
    
    for i, benefit in enumerate(benefits):
        ax.text(5, 95 - i*3, benefit, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('lora_adapters.png', dpi=150, bbox_inches='tight')
    print("[OK] Сохранено: lora_adapters.png")
    plt.show()


def draw_box(ax, x, y, width, height, text, color, label, linewidth=1, linestyle='-'):
    """Вспомогательная функция для рисования блоков"""
    rect = patches.Rectangle((x, y), width, height, facecolor=color, 
                            edgecolor='black', linewidth=linewidth, 
                            linestyle=linestyle, alpha=0.8)
    ax.add_patch(rect)
    
    # Разбить текст на строки
    lines = text.split('\n')
    for i, line in enumerate(lines):
        ax.text(x + width/2, y + height - 1 - i*0.8, line, 
               ha='center', va='top', fontsize=8, wrap=True)
    
    # Метка
    ax.text(x + width/2, y - 0.8, label, ha='center', va='top', 
           fontsize=9, fontweight='bold', color='red')


def main():
    """Запуск всех визуализаций"""
    print("=" * 60)
    print("Визуализация архитектуры трансформера GPT-2")
    print("=" * 60)
    
    try:
        print("\n1. Архитектура трансформера...")
        draw_transformer_architecture()
        
        print("\n2. Механизм внимания...")
        draw_attention_mechanism()
        
        print("\n3. Цикл обучения...")
        draw_training_loop()
        
        print("\n4. LoRA адаптеры...")
        draw_lora_adapters()
        
        print("\n" + "=" * 60)
        print("✓ Все визуализации созданы!")
        print("=" * 60)
        print("\nФайлы:")
        print("  - transformer_architecture.png")
        print("  - attention_mechanism.png")
        print("  - training_loop.png")
        print("  - lora_adapters.png")
        print("\nОткройте их для просмотра!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
