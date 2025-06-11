import time
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple
import threading
import queue

class TrainingController:
    def __init__(self):
        self.start_time = None
        self.target_duration = None  # En minutos
        self.target_episodes = None
        self.target_win_rate = None
        self.current_episode = 0
        self.training_active = False
        self.pause_training = False
        self.stop_training = False
        self.training_stats = []
        self.evaluation_results = []
        self.best_models_saved = []
        
        # Cola para comunicación entre hilos
        self.command_queue = queue.Queue()
        self.status_queue = queue.Queue()
        
    def set_training_target(self, duration_minutes=None, episodes=None, target_win_rate=None):
        """Establece objetivos de entrenamiento"""
        self.target_duration = duration_minutes
        self.target_episodes = episodes
        self.target_win_rate = target_win_rate
        
        print(f"Objetivos de entrenamiento establecidos:")
        if duration_minutes:
            print(f"  - Duración máxima: {duration_minutes} minutos")
        if episodes:
            print(f"  - Episodios máximos: {episodes}")
        if target_win_rate:
            print(f"  - Tasa de victoria objetivo: {target_win_rate:.2%}")
    
    def start_training(self):
        """Inicia el entrenamiento"""
        self.start_time = time.time()
        self.training_active = True
        self.pause_training = False
        self.stop_training = False
        self.current_episode = 0
        print(f"Entrenamiento iniciado a las {datetime.now().strftime('%H:%M:%S')}")
    
    def pause_training_session(self):
        """Pausa el entrenamiento"""
        self.pause_training = True
        print("Entrenamiento pausado")
    
    def resume_training(self):
        """Reanuda el entrenamiento"""
        self.pause_training = False
        print("Entrenamiento reanudado")
    
    def stop_training_session(self):
        """Detiene el entrenamiento"""
        self.stop_training = True
        self.training_active = False
        print("Entrenamiento detenido")
    
    def should_continue_training(self) -> bool:
        """Verifica si el entrenamiento debe continuar"""
        if self.stop_training:
            return False
        
        if self.pause_training:
            return True  # Pausado pero no detenido
        
        # Verificar duración
        if self.target_duration and self.start_time:
            elapsed_minutes = (time.time() - self.start_time) / 60
            if elapsed_minutes >= self.target_duration:
                print(f"Objetivo de duración alcanzado: {elapsed_minutes:.1f} minutos")
                return False
        
        # Verificar episodios
        if self.target_episodes and self.current_episode >= self.target_episodes:
            print(f"Objetivo de episodios alcanzado: {self.current_episode}")
            return False
        
        # Verificar tasa de victoria
        if self.target_win_rate and len(self.evaluation_results) > 0:
            latest_win_rate = self.evaluation_results[-1].get('avg_win_rate', 0)
            if latest_win_rate >= self.target_win_rate:
                print(f"Objetivo de tasa de victoria alcanzado: {latest_win_rate:.2%}")
                return False
        
        return True
    
    def update_episode(self, episode: int):
        """Actualiza el episodio actual"""
        self.current_episode = episode
    
    def add_training_stats(self, stats: Dict):
        """Añade estadísticas de entrenamiento"""
        stats['timestamp'] = time.time()
        stats['episode'] = self.current_episode
        if self.start_time:
            stats['elapsed_minutes'] = (time.time() - self.start_time) / 60
        self.training_stats.append(stats)
    
    def add_evaluation_result(self, result: Dict):
        """Añade resultado de evaluación"""
        result['timestamp'] = time.time()
        result['episode'] = self.current_episode
        if self.start_time:
            result['elapsed_minutes'] = (time.time() - self.start_time) / 60
        self.evaluation_results.append(result)
    
    def get_training_progress(self) -> Dict:
        """Obtiene el progreso actual del entrenamiento"""
        progress = {
            'active': self.training_active,
            'paused': self.pause_training,
            'current_episode': self.current_episode,
            'elapsed_time': 0,
            'progress_percentage': 0
        }
        
        if self.start_time:
            progress['elapsed_time'] = time.time() - self.start_time
            
            # Calcular porcentaje de progreso
            if self.target_duration:
                elapsed_minutes = progress['elapsed_time'] / 60
                progress['progress_percentage'] = min(100, (elapsed_minutes / self.target_duration) * 100)
            elif self.target_episodes:
                progress['progress_percentage'] = min(100, (self.current_episode / self.target_episodes) * 100)
        
        return progress
    
    def save_training_session(self, filename=None):
        """Guarda la sesión de entrenamiento"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_session_{timestamp}.json"
        
        session_data = {
            'start_time': self.start_time,
            'target_duration': self.target_duration,
            'target_episodes': self.target_episodes,
            'target_win_rate': self.target_win_rate,
            'final_episode': self.current_episode,
            'training_stats': self.training_stats,
            'evaluation_results': self.evaluation_results,
            'best_models_saved': self.best_models_saved
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"Sesión de entrenamiento guardada en: {filename}")
        return filename
    
    def load_training_session(self, filename):
        """Carga una sesión de entrenamiento"""
        with open(filename, 'r') as f:
            session_data = json.load(f)
        
        self.start_time = session_data.get('start_time')
        self.target_duration = session_data.get('target_duration')
        self.target_episodes = session_data.get('target_episodes')
        self.target_win_rate = session_data.get('target_win_rate')
        self.current_episode = session_data.get('final_episode', 0)
        self.training_stats = session_data.get('training_stats', [])
        self.evaluation_results = session_data.get('evaluation_results', [])
        self.best_models_saved = session_data.get('best_models_saved', [])
        
        print(f"Sesión de entrenamiento cargada desde: {filename}")


class LearningEvaluator:
    def __init__(self):
        self.baseline_performance = None
        self.evaluation_history = []
        self.learning_metrics = {}
        
    def set_baseline(self, win_rate_eq1: float, win_rate_eq2: float, avg_game_length: float):
        """Establece la línea base de rendimiento"""
        self.baseline_performance = {
            'win_rate_eq1': win_rate_eq1,
            'win_rate_eq2': win_rate_eq2,
            'avg_game_length': avg_game_length,
            'timestamp': time.time()
        }
        print(f"Línea base establecida:")
        print(f"  - Tasa victoria Equipo 1: {win_rate_eq1:.2%}")
        print(f"  - Tasa victoria Equipo 2: {win_rate_eq2:.2%}")
        print(f"  - Duración promedio juego: {avg_game_length:.1f} turnos")
    
    def evaluate_learning_progress(self, agents, num_games=100) -> Dict:
        """Evalúa el progreso de aprendizaje de los agentes"""
        from mus_env import mus
        
        print(f"Evaluando progreso de aprendizaje con {num_games} juegos...")
        
        # Métricas a evaluar
        wins = {"equipo_1": 0, "equipo_2": 0}
        game_lengths = []
        decision_quality = {"good_decisions": 0, "total_decisions": 0}
        phase_performance = {
            "MUS": {"optimal": 0, "total": 0},
            "GRANDE": {"optimal": 0, "total": 0},
            "CHICA": {"optimal": 0, "total": 0},
            "PARES": {"optimal": 0, "total": 0},
            "JUEGO": {"optimal": 0, "total": 0},
            "RECUENTO": {"optimal": 0, "total": 0}
        }
        
        for game in range(num_games):
            env = mus.env()
            env.reset()
            
            game_length = 0
            states = {}
            
            # Inicializar estados
            for agent in env.agents:
                obs = env.observe(agent)
                states[agent] = self._process_observation(obs)
            
            while not all(env.dones.values()) and game_length < 1000:
                current_agent = env.agent_selection
                valid_actions = self._get_valid_actions(env, current_agent)
                
                # Evaluar calidad de decisión
                optimal_action = self._get_optimal_action(env, current_agent, states[current_agent])
                
                # Usar epsilon bajo para evaluación
                old_epsilon = agents[current_agent].epsilon
                agents[current_agent].epsilon = 0.05
                
                action = agents[current_agent].act(states[current_agent], valid_actions)
                
                # Restaurar epsilon
                agents[current_agent].epsilon = old_epsilon
                
                # Evaluar decisión
                if action == optimal_action:
                    decision_quality["good_decisions"] += 1
                    phase_performance[env.fase_actual]["optimal"] += 1
                
                decision_quality["total_decisions"] += 1
                phase_performance[env.fase_actual]["total"] += 1
                
                env.step(action)
                game_length += 1
                
                # Actualizar estado
                if not env.dones.get(current_agent, False):
                    next_obs = env.observe(current_agent)
                    states[current_agent] = self._process_observation(next_obs)
            
            # Registrar resultado
            game_lengths.append(game_length)
            if env.puntos_equipos["equipo_1"] >= 30:
                wins["equipo_1"] += 1
            elif env.puntos_equipos["equipo_2"] >= 30:
                wins["equipo_2"] += 1
        
        # Calcular métricas
        total_games = wins["equipo_1"] + wins["equipo_2"]
        win_rate_eq1 = wins["equipo_1"] / total_games if total_games > 0 else 0
        win_rate_eq2 = wins["equipo_2"] / total_games if total_games > 0 else 0
        avg_game_length = np.mean(game_lengths) if game_lengths else 0
        decision_accuracy = decision_quality["good_decisions"] / decision_quality["total_decisions"] if decision_quality["total_decisions"] > 0 else 0
        
        # Calcular rendimiento por fase
        phase_accuracy = {}
        for phase, stats in phase_performance.items():
            if stats["total"] > 0:
                phase_accuracy[phase] = stats["optimal"] / stats["total"]
            else:
                phase_accuracy[phase] = 0
        
        evaluation_result = {
            'timestamp': time.time(),
            'win_rate_eq1': win_rate_eq1,
            'win_rate_eq2': win_rate_eq2,
            'avg_game_length': avg_game_length,
            'decision_accuracy': decision_accuracy,
            'phase_accuracy': phase_accuracy,
            'total_games_evaluated': num_games,
            'games_completed': total_games
        }
        
        # Comparar con línea base si existe
        if self.baseline_performance:
            evaluation_result['improvement'] = {
                'win_rate_eq1': win_rate_eq1 - self.baseline_performance['win_rate_eq1'],
                'win_rate_eq2': win_rate_eq2 - self.baseline_performance['win_rate_eq2'],
                'avg_game_length': avg_game_length - self.baseline_performance['avg_game_length']
            }
        
        self.evaluation_history.append(evaluation_result)
        
        print(f"Evaluación completada:")
        print(f"  - Tasa victoria Eq1: {win_rate_eq1:.2%}")
        print(f"  - Tasa victoria Eq2: {win_rate_eq2:.2%}")
        print(f"  - Precisión decisiones: {decision_accuracy:.2%}")
        print(f"  - Duración promedio: {avg_game_length:.1f} turnos")
        
        return evaluation_result
    
    def _process_observation(self, obs):
        """Procesa observación para evaluación"""
        return np.concatenate([
            obs["cartas"].flatten(),
            np.eye(7)[obs["fase"]],
            np.eye(4)[obs["turno"]],
            [obs.get("apuesta_actual", 0) / 30.0],
            [obs.get("equipo_apostador", 0) / 2.0]
        ])
    
    def _get_valid_actions(self, env, agent):
        """Obtiene acciones válidas"""
        if env.fase_actual == "MUS":
            return [2, 3]
        elif env.fase_actual == "DESCARTE":
            return [4] + list(range(11, 15))
        elif env.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
            if agent not in env.jugadores_que_pueden_hablar:
                return [0]
            valid = [0, 1]
            if env.apuesta_actual > 0:
                equipo_actual = env.equipo_de_jugador[agent]
                if equipo_actual != env.equipo_apostador:
                    valid.extend([5, 7])
            if not hasattr(env, 'hay_ordago') or not env.hay_ordago:
                valid.append(6)
            return valid
        return [0]
    
    def _get_optimal_action(self, env, agent, state):
        """Determina la acción óptima (heurística simple)"""
        if env.fase_actual == "MUS":
            # Heurística: pedir mus si hay cartas malas
            mano = env.manos[agent]
            valores = [carta[0] for carta in mano]
            cartas_malas = sum(1 for v in valores if 5 <= v <= 7)
            return 2 if cartas_malas >= 2 else 3
        
        elif env.fase_actual == "DESCARTE":
            return 4  # Siempre confirmar por simplicidad
        
        elif env.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
            if agent not in env.jugadores_que_pueden_hablar:
                return 0
            
            # Heurística simple: apostar si tienes buena mano
            mano = env.manos[agent]
            if env.fase_actual == "GRANDE":
                figuras = sum(1 for carta in mano if carta[0] >= 10)
                return 1 if figuras >= 2 else 0
            elif env.fase_actual == "CHICA":
                cartas_bajas = sum(1 for carta in mano if carta[0] <= 4)
                return 1 if cartas_bajas >= 2 else 0
            elif env.fase_actual == "PARES":
                return 1 if env.declaraciones_pares.get(agent, False) else 0
            elif env.fase_actual == "JUEGO":
                return 1 if env.declaraciones_juego.get(agent, False) else 0
        
        return 0
    
    def has_learned_significantly(self, min_improvement=0.1) -> bool:
        """Verifica si los agentes han aprendido significativamente"""
        if not self.baseline_performance or len(self.evaluation_history) == 0:
            return False
        
        latest_eval = self.evaluation_history[-1]
        
        # Verificar mejora en tasa de victoria
        win_rate_improvement = latest_eval.get('improvement', {}).get('win_rate_eq1', 0)
        decision_accuracy = latest_eval.get('decision_accuracy', 0)
        
        # Criterios de aprendizaje significativo
        learned = (
            abs(win_rate_improvement) >= min_improvement or  # Mejora en tasa de victoria
            decision_accuracy >= 0.7  # Buena precisión en decisiones
        )
        
        return learned
    
    def generate_learning_report(self) -> str:
        """Genera un reporte de aprendizaje"""
        if len(self.evaluation_history) == 0:
            return "No hay datos de evaluación disponibles."
        
        latest = self.evaluation_history[-1]
        report = []
        
        report.append("=== REPORTE DE APRENDIZAJE ===")
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Evaluaciones realizadas: {len(self.evaluation_history)}")
        
        report.append("\n--- RENDIMIENTO ACTUAL ---")
        report.append(f"Tasa de victoria Equipo 1: {latest['win_rate_eq1']:.2%}")
        report.append(f"Tasa de victoria Equipo 2: {latest['win_rate_eq2']:.2%}")
        report.append(f"Precisión en decisiones: {latest['decision_accuracy']:.2%}")
        report.append(f"Duración promedio de juego: {latest['avg_game_length']:.1f} turnos")
        
        if self.baseline_performance and 'improvement' in latest:
            report.append("\n--- MEJORA RESPECTO A LÍNEA BASE ---")
            imp = latest['improvement']
            report.append(f"Mejora tasa victoria Eq1: {imp['win_rate_eq1']:+.2%}")
            report.append(f"Mejora tasa victoria Eq2: {imp['win_rate_eq2']:+.2%}")
            report.append(f"Cambio duración juego: {imp['avg_game_length']:+.1f} turnos")
        
        report.append("\n--- RENDIMIENTO POR FASE ---")
        for phase, accuracy in latest['phase_accuracy'].items():
            report.append(f"{phase}: {accuracy:.2%}")
        
        # Tendencia de aprendizaje
        if len(self.evaluation_history) >= 3:
            report.append("\n--- TENDENCIA DE APRENDIZAJE ---")
            recent_evals = self.evaluation_history[-3:]
            decision_trend = [e['decision_accuracy'] for e in recent_evals]
            
            if decision_trend[-1] > decision_trend[0]:
                report.append("✅ Tendencia de mejora en precisión de decisiones")
            else:
                report.append("⚠️ Tendencia de deterioro en precisión de decisiones")
        
        # Conclusión
        report.append("\n--- CONCLUSIÓN ---")
        if self.has_learned_significantly():
            report.append("✅ Los agentes HAN APRENDIDO significativamente")
        else:
            report.append("❌ Los agentes AÚN NO han aprendido significativamente")
        
        return "\n".join(report)
    
    def plot_learning_curves(self, save_path=None):
        """Genera gráficos de curvas de aprendizaje"""
        if len(self.evaluation_history) < 2:
            print("Insuficientes datos para generar gráficos")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Curvas de Aprendizaje - Agentes Mus', fontsize=16)
        
        episodes = [e.get('episode', i) for i, e in enumerate(self.evaluation_history)]
        
        # Tasa de victoria
        win_rates_eq1 = [e['win_rate_eq1'] for e in self.evaluation_history]
        win_rates_eq2 = [e['win_rate_eq2'] for e in self.evaluation_history]
        
        axes[0, 0].plot(episodes, win_rates_eq1, 'b-', label='Equipo 1', marker='o')
        axes[0, 0].plot(episodes, win_rates_eq2, 'r-', label='Equipo 2', marker='s')
        axes[0, 0].set_title('Tasa de Victoria por Equipo')
        axes[0, 0].set_xlabel('Episodio')
        axes[0, 0].set_ylabel('Tasa de Victoria')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Precisión de decisiones
        decision_accuracy = [e['decision_accuracy'] for e in self.evaluation_history]
        axes[0, 1].plot(episodes, decision_accuracy, 'g-', marker='o')
        axes[0, 1].set_title('Precisión en Decisiones')
        axes[0, 1].set_xlabel('Episodio')
        axes[0, 1].set_ylabel('Precisión')
        axes[0, 1].grid(True)
        
        # Duración de juegos
        game_lengths = [e['avg_game_length'] for e in self.evaluation_history]
        axes[1, 0].plot(episodes, game_lengths, 'm-', marker='o')
        axes[1, 0].set_title('Duración Promedio de Juegos')
        axes[1, 0].set_xlabel('Episodio')
        axes[1, 0].set_ylabel('Turnos')
        axes[1, 0].grid(True)
        
        # Rendimiento por fase (última evaluación)
        if self.evaluation_history:
            latest = self.evaluation_history[-1]
            phases = list(latest['phase_accuracy'].keys())
            accuracies = list(latest['phase_accuracy'].values())
            
            axes[1, 1].bar(phases, accuracies, color=['blue', 'green', 'red', 'orange', 'purple'])
            axes[1, 1].set_title('Precisión por Fase (Última Evaluación)')
            axes[1, 1].set_xlabel('Fase')
            axes[1, 1].set_ylabel('Precisión')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráficos guardados en: {save_path}")
        
        plt.show()


def create_training_interface():
    """Crea una interfaz simple para controlar el entrenamiento"""
    controller = TrainingController()
    evaluator = LearningEvaluator()
    
    def show_menu():
        print("\n" + "="*50)
        print("CONTROL DE ENTRENAMIENTO MARL - MUS")
        print("="*50)
        print("1. Configurar objetivos de entrenamiento")
        print("2. Iniciar entrenamiento")
        print("3. Pausar/Reanudar entrenamiento")
        print("4. Detener entrenamiento")
        print("5. Ver progreso actual")
        print("6. Evaluar aprendizaje")
        print("7. Generar reporte de aprendizaje")
        print("8. Mostrar gráficos de aprendizaje")
        print("9. Guardar sesión")
        print("10. Cargar sesión")
        print("0. Salir")
        print("-"*50)
    
    while True:
        show_menu()
        choice = input("Selecciona una opción: ").strip()
        
        if choice == "1":
            print("\nConfiguración de objetivos:")
            duration = input("Duración máxima en minutos (Enter para omitir): ").strip()
            episodes = input("Número máximo de episodios (Enter para omitir): ").strip()
            win_rate = input("Tasa de victoria objetivo 0-1 (Enter para omitir): ").strip()
            
            controller.set_training_target(
                duration_minutes=int(duration) if duration else None,
                episodes=int(episodes) if episodes else None,
                target_win_rate=float(win_rate) if win_rate else None
            )
        
        elif choice == "2":
            controller.start_training()
        
        elif choice == "3":
            if controller.pause_training:
                controller.resume_training()
            else:
                controller.pause_training_session()
        
        elif choice == "4":
            controller.stop_training_session()
        
        elif choice == "5":
            progress = controller.get_training_progress()
            print(f"\nProgreso actual:")
            print(f"  Estado: {'Activo' if progress['active'] else 'Inactivo'}")
            if progress['paused']:
                print(f"  Estado: PAUSADO")
            print(f"  Episodio actual: {progress['current_episode']}")
            print(f"  Tiempo transcurrido: {progress['elapsed_time']/60:.1f} minutos")
            print(f"  Progreso: {progress['progress_percentage']:.1f}%")
        
        elif choice == "6":
            print("Esta opción requiere agentes entrenados cargados.")
            print("Ejecuta desde el script de entrenamiento para evaluación completa.")
        
        elif choice == "7":
            report = evaluator.generate_learning_report()
            print(f"\n{report}")
        
        elif choice == "8":
            evaluator.plot_learning_curves()
        
        elif choice == "9":
            filename = controller.save_training_session()
            print(f"Sesión guardada: {filename}")
        
        elif choice == "10":
            filename = input("Nombre del archivo de sesión: ").strip()
            try:
                controller.load_training_session(filename)
            except FileNotFoundError:
                print("Archivo no encontrado")
        
        elif choice == "0":
            break
        
        else:
            print("Opción no válida")

if __name__ == "__main__":
    create_training_interface()