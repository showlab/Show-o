import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


class MoEVisualizer:
    def __init__(self, num_experts: int, layer_id: Optional[int] = None):
        self.num_experts = num_experts
        self.layer_id = layer_id
    
    def _set_no_data_plot(self, ax, message: str = "No data", title: str = None, 
                          xlabel: str = None, ylabel: str = None, 
                          message_fontsize: int = 14, title_fontsize: int = 12):
        """Устанавливает текст 'No data' на оси с заданными параметрами"""
        ax.text(0.5, 0.5, message, ha='center', va='center', 
                transform=ax.transAxes, fontsize=message_fontsize)
        if title:
            ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)
    
    def create_distribution_heatmap(self, distribution_history: Dict[int, Dict[int, int]], 
                                   modality_name: str = "overall", 
                                   global_step: int = 0,
                                   alpha_value: Optional[float] = None) -> bytes:
        """Создает heatmap распределений гейтов"""
        if not distribution_history:
            # Создаем пустой график с сообщением "No data"
            fig, ax = plt.subplots(figsize=(14, 8))
            self._set_no_data_plot(ax, message='No data', 
                                  title=f'Gate Distribution ({modality_name})',
                                  xlabel='Iteration', ylabel='Expert ID',
                                  message_fontsize=20, title_fontsize=14)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return buf.getvalue()
        
        steps = sorted(distribution_history.keys())
        if not steps:
            # Создаем пустой график с сообщением "No data"
            fig, ax = plt.subplots(figsize=(14, 8))
            self._set_no_data_plot(ax, message='No data', 
                                  title=f'Gate Distribution ({modality_name})',
                                  xlabel='Iteration', ylabel='Expert ID',
                                  message_fontsize=20, title_fontsize=14)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return buf.getvalue()
        
        all_expert_ids = set()
        for step_counts in distribution_history.values():
            all_expert_ids.update(step_counts.keys())
        expert_ids = sorted(all_expert_ids)
        matrix = np.zeros((len(expert_ids), len(steps)))
        
        for col_idx, step in enumerate(steps):
            step_counts = distribution_history[step]
            total = sum(step_counts.values())
            if total > 0:
                for row_idx, expert_id in enumerate(expert_ids):
                    matrix[row_idx, col_idx] = step_counts.get(expert_id, 0) / total
        
        fig, ax = plt.subplots(figsize=(14, 8))
        title_suffix = f" — constant α={alpha_value}" if alpha_value is not None else ""
        title = f"Gate Distribution{title_suffix}"
        if modality_name != "overall":
            title = f"Gate Distribution ({modality_name}){title_suffix}"
        
        im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Expert ID', fontsize=12)
        
        step_indices = np.arange(len(steps))
        if len(steps) > 20:
            tick_step = max(1, len(steps) // 20)
            ax.set_xticks(step_indices[::tick_step])
            ax.set_xticklabels([steps[i] for i in step_indices[::tick_step]], rotation=45)
        else:
            ax.set_xticks(step_indices)
            ax.set_xticklabels(steps, rotation=45)
        
        ax.set_yticks(np.arange(len(expert_ids)))
        ax.set_yticklabels(expert_ids)
        
        # Добавляем colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Gate Distribution', fontsize=11)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    
    def create_single_heatmap(self, ax, distribution_history: Dict[int, Dict[int, int]], 
                              modality_name: str = "overall", 
                              alpha_value: Optional[float] = None) -> bool:
        """Создает одну heatmap на переданной оси. Если данных нет, возвращает False"""
        if not distribution_history:
            return False
        
        steps = sorted(distribution_history.keys())
        if not steps:
            return False
        
        all_expert_ids = set()
        for step_counts in distribution_history.values():
            all_expert_ids.update(step_counts.keys())
        expert_ids = sorted(all_expert_ids)
        
        if len(expert_ids) == 0:
            return False
        
        matrix = np.zeros((len(expert_ids), len(steps)))
        has_data = False
        
        for col_idx, step in enumerate(steps):
            step_counts = distribution_history[step]
            total = sum(step_counts.values())
            if total > 0:
                has_data = True
                for row_idx, expert_id in enumerate(expert_ids):
                    matrix[row_idx, col_idx] = step_counts.get(expert_id, 0) / total
        
        if not has_data or matrix.max() == 0:
            return False
        
        title_suffix = f" — constant α={alpha_value}" if alpha_value is not None else ""
        title = f"Gate Distribution ({modality_name}){title_suffix}"
        
        im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Expert ID', fontsize=10)
        
        step_indices = np.arange(len(steps))
        if len(steps) > 20:
            tick_step = max(1, len(steps) // 20)
            ax.set_xticks(step_indices[::tick_step])
            ax.set_xticklabels([steps[i] for i in step_indices[::tick_step]], rotation=45, fontsize=8)
        else:
            ax.set_xticks(step_indices)
            ax.set_xticklabels(steps, rotation=45, fontsize=8)
        
        ax.set_yticks(np.arange(len(expert_ids)))
        ax.set_yticklabels(expert_ids, fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Gate Distribution')
        return True
    
    def create_expert_activation_histogram(self, expert_counts: Dict[int, int], 
                                          modality_name: Optional[str] = None, 
                                          ax: Optional[plt.Axes] = None) -> Optional[bytes]:
        """Создает гистограмму активаций экспертов"""
        experts = list(expert_counts.keys())
        activations = list(expert_counts.values())
        
        create_new_figure = (ax is None)
        if create_new_figure:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
        
        # Создаем столбчатую диаграмму
        bars = ax.bar(experts, activations, alpha=0.7, color='green', edgecolor='black')
        title = f'Expert Token Counts'
        if modality_name:
            title += f' ({modality_name})'
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Expert ID', fontsize=10)
        ax.set_ylabel('Number of Activations', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, count in zip(bars, activations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(count), ha='center', va='bottom', fontsize=8)
        
        # Добавляем статистики
        total_activations = sum(activations)
        balance = max(activations) - min(activations) if activations else 0
        ax.text(0.02, 0.98, f'Total: {total_activations}\nBalance: {balance}', 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Если создали новый figure, возвращаем bytes
        if create_new_figure:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return buf.getvalue()
    
    def create_modality_combined_plot(self, text_history: Optional[Dict[int, Dict[int, int]]] = None,
                                     image_history: Optional[Dict[int, Dict[int, int]]] = None,
                                     text_expert_counts: Optional[Dict[int, int]] = None,
                                     image_expert_counts: Optional[Dict[int, int]] = None,
                                     global_step: int = 0,
                                     alpha_value: Optional[float] = None) -> bytes:
        """Создает объединенный plot с 4 subplot'ами: 2 heatmap (text, image) и 2 гистограммы (text, image)"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'Gate Distribution Analysis - Layer {self.layer_id} - Step {global_step}', 
                     fontsize=16, fontweight='bold')
        
        # Верхний ряд: Heatmap для text и image
        # Heatmap для text (верхний левый)
        if text_history and len(text_history) > 0:
            success = self.create_single_heatmap(axes[0, 0], text_history, "text", alpha_value)
            if not success:
                self._set_no_data_plot(axes[0, 0], message='No text data',
                                      title='Gate Distribution (text)',
                                      xlabel='Iteration', ylabel='Expert ID')
        else:
            self._set_no_data_plot(axes[0, 0], message='No text data',
                                  title='Gate Distribution (text)',
                                  xlabel='Iteration', ylabel='Expert ID')
        
        # Heatmap для image (верхний правый)
        if image_history and len(image_history) > 0:
            success = self.create_single_heatmap(axes[0, 1], image_history, "image", alpha_value)
            if not success:
                self._set_no_data_plot(axes[0, 1], message='No image data',
                                      title='Gate Distribution (image)',
                                      xlabel='Iteration', ylabel='Expert ID')
        else:
            self._set_no_data_plot(axes[0, 1], message='No image data',
                                  title='Gate Distribution (image)',
                                  xlabel='Iteration', ylabel='Expert ID')
        
        # Нижний ряд: Гистограммы активаций экспертов
        # Гистограмма для text (нижний левый)
        if text_expert_counts and len(text_expert_counts) > 0:
            bars = axes[1, 0].bar(list(text_expert_counts.keys()), list(text_expert_counts.values()), 
                          alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].set_title('Expert Token Counts (text)', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Expert ID', fontsize=10)
            axes[1, 0].set_ylabel('Number of Activations', fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
            # Добавляем значения на столбцы
            for bar, count in zip(bars, text_expert_counts.values()):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        str(count), ha='center', va='bottom', fontsize=8)
        else:
            self._set_no_data_plot(axes[1, 0], message='No text data',
                                  title='Expert Token Counts (text)',
                                  xlabel='Expert ID', ylabel='Number of Activations')
        
        # Гистограмма для image (нижний правый)
        if image_expert_counts and len(image_expert_counts) > 0:
            bars = axes[1, 1].bar(list(image_expert_counts.keys()), list(image_expert_counts.values()), 
                          alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].set_title('Expert Token Counts (image)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Expert ID', fontsize=10)
            axes[1, 1].set_ylabel('Number of Activations', fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
            # Добавляем значения на столбцы
            for bar, count in zip(bars, image_expert_counts.values()):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        str(count), ha='center', va='bottom', fontsize=8)
        else:
            self._set_no_data_plot(axes[1, 1], message='No image data',
                                  title='Expert Token Counts (image)',
                                  xlabel='Expert ID', ylabel='Number of Activations')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    
    def create_domain_plot(self, domain_id: str, domain_history: Dict[int, Dict[int, int]], 
                          global_step: int, current_expert_counts: Optional[Dict[int, int]] = None) -> bytes:
        """Создает объединенный plot для домена: heatmap (сверху) и histogram (снизу) на одном графике"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle(f'Gate Distribution Analysis - {domain_id} - Layer {self.layer_id} - Step {global_step}', 
                     fontsize=16, fontweight='bold')
        
        # Верхний subplot: Heatmap
        if domain_history and len(domain_history) > 0:
            success = self.create_single_heatmap(axes[0], domain_history, domain_id)
            if not success:
                self._set_no_data_plot(axes[0], message='No data',
                                      title=f'Gate Distribution ({domain_id})',
                                      xlabel='Iteration', ylabel='Expert ID')
        else:
            self._set_no_data_plot(axes[0], message='No data',
                                  title=f'Gate Distribution ({domain_id})',
                                  xlabel='Iteration', ylabel='Expert ID')
        
        # Нижний subplot: Histogram
        if current_expert_counts and len(current_expert_counts) > 0:
            bars = axes[1].bar(list(current_expert_counts.keys()), list(current_expert_counts.values()), 
                              alpha=0.7, color='purple', edgecolor='black')
            axes[1].set_title(f'Expert Token Counts ({domain_id})', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Expert ID', fontsize=10)
            axes[1].set_ylabel('Number of Activations', fontsize=10)
            axes[1].grid(True, alpha=0.3)
            # Добавляем значения на столбцы
            for bar, count in zip(bars, current_expert_counts.values()):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        str(count), ha='center', va='bottom', fontsize=8)
        else:
            self._set_no_data_plot(axes[1], message='No data',
                                  title=f'Expert Token Counts ({domain_id})',
                                  xlabel='Expert ID', ylabel='Number of Activations')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    

    def create_all_domains_combined_plot(self, domain_distribution_history: Dict[str, Dict[int, Dict[int, int]]],
                                        global_step: int) -> bytes:
        """Создает общий график всех доменов: для каждого домена heatmap и распределение"""
        all_domains = list(domain_distribution_history.keys())
        
        if not all_domains:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            self._set_no_data_plot(ax, message='No domain data available',
                                  title='Gate Distribution by Domain - All Domains',
                                  message_fontsize=16, title_fontsize=14)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return buf.getvalue()
        
        all_domains = sorted(all_domains)
        n_domains = len(all_domains)
        
        fig, axes = plt.subplots(n_domains, 2, figsize=(12, 4 * n_domains))
        fig.suptitle(f'Gate Distribution by Domain - Layer {self.layer_id} - Step {global_step}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        if n_domains == 1:
            axes = axes.reshape(1, -1)
        
        for idx, domain_id in enumerate(all_domains):
            ax_heatmap = axes[idx, 0]
            ax_dist = axes[idx, 1]
            
            domain_history = domain_distribution_history.get(domain_id, {})
            
            if domain_history:
                steps = sorted(domain_history.keys())
                if steps:
                    recent_steps = steps[-min(10, len(steps)):]
                    
                    heatmap_data = np.zeros((self.num_experts, len(recent_steps)))
                    
                    for step_idx, step in enumerate(recent_steps):
                        step_data = domain_history[step]
                        for expert_id, count in step_data.items():
                            if expert_id < self.num_experts:
                                heatmap_data[expert_id, step_idx] = count
                    
                    if heatmap_data.sum() > 0:
                        heatmap_data_norm = heatmap_data / (heatmap_data.sum(axis=0, keepdims=True) + 1e-8)
                    else:
                        heatmap_data_norm = heatmap_data
                    
                    im = ax_heatmap.imshow(heatmap_data_norm, aspect='auto', cmap='viridis', 
                                          interpolation='nearest', vmin=0, vmax=1)
                    ax_heatmap.set_title(f'{domain_id} - Heatmap', fontsize=11, fontweight='bold')
                    ax_heatmap.set_xlabel('Step (recent)', fontsize=9)
                    ax_heatmap.set_ylabel('Expert ID', fontsize=9)
                    ax_heatmap.set_yticks(range(self.num_experts))
                    ax_heatmap.set_yticklabels(range(self.num_experts))
                    if len(recent_steps) <= 10:
                        ax_heatmap.set_xticks(range(len(recent_steps)))
                        ax_heatmap.set_xticklabels([str(s) for s in recent_steps], rotation=45)
                    plt.colorbar(im, ax=ax_heatmap, label='Normalized Gate Distribution')
                else:
                    self._set_no_data_plot(ax_heatmap, message='No data',
                                          title=f'{domain_id} - Heatmap',
                                          message_fontsize=12, title_fontsize=11)
            else:
                self._set_no_data_plot(ax_heatmap, message='No data',
                                      title=f'{domain_id} - Heatmap',
                                      message_fontsize=12, title_fontsize=11)
            
            current_expert_counts = None
            if domain_history:
                current_expert_counts = domain_history.get(global_step, {})
            
            if current_expert_counts and len(current_expert_counts) > 0:
                expert_ids = list(current_expert_counts.keys())
                counts = list(current_expert_counts.values())
                bars = ax_dist.bar(expert_ids, counts, alpha=0.7, color='steelblue', edgecolor='black')
                ax_dist.set_title(f'{domain_id} - Distribution (Step {global_step})', 
                                fontsize=11, fontweight='bold')
                ax_dist.set_xlabel('Expert ID', fontsize=9)
                ax_dist.set_ylabel('Number of Activations', fontsize=9)
                ax_dist.grid(True, alpha=0.3)
                
                for bar, count in zip(bars, counts):
                    if count > 0:
                        ax_dist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01, 
                                     str(count), ha='center', va='bottom', fontsize=8)
            else:
                self._set_no_data_plot(ax_dist, message='No data',
                                      title=f'{domain_id} - Distribution (Step {global_step})',
                                      xlabel='Expert ID', ylabel='Number of Activations',
                                      message_fontsize=12, title_fontsize=11)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
