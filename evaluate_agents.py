import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mus_env import mus
from marl_agent import MARLAgent
import torch
import json
from datetime import datetime
import pandas as pd

class AgentEvaluator:
    def __init__(self):
        self.evaluation_results = []
        
    def load_agents(self, model_prefix="model_jugador"):
        """Carga los agentes entrenados"""
        agents = {}
        state_size = 21
        action_size = 15
        
        for i in range(4):
            team = "equipo_1" if i in [0, 2] else "equipo_2"
            agents[f"jugador_{i}"] = MARLAgent(state_size, action_size, i, team)
            
            # Intentar cargar diferentes versiones del modelo
            model_paths = [
                f"best_model_jugador_{i}.pth",
                f"{model_prefix}_{i}_final.pth",
                f"{model_prefix}_{i}.pth"
            ]
            
            loaded = False
            for path in model_paths:
                try:
                    agents[f"jugador_{i}"].load(path)
                    print(f"✅ Modelo cargado para jugador_{i}: {path}")
                    loaded = True
                    break
                except:
                    continue
            
            if not loaded:
                print(f"⚠️ No se pudo cargar modelo para jugador_{i}, usando modelo aleatorio")
        
        return agents
    
    def process_observation(self, obs):
        """Procesa observación para los agentes"""
        try:
            cartas_flat = obs["cartas"].flatten()
            fase_onehot = np.zeros(7)
            fase_onehot[obs["fase"]] = 1
            turno_onehot = np.zeros(4)
            turno_onehot[obs["turno"]] = 1
            
            apuesta_norm = obs.get("apuesta_actual", 0) / 30.0
            equipo_apostador = obs.get("equipo_apostador", 0) / 2.0
            
            state = np.concatenate([
                cartas_flat, fase_onehot, turno_onehot, [apuesta_norm], [equipo_apostador]
            ])
            return state
        except Exception as e:
            print(f"Error procesando observación: {e}")
            return np.zeros(21)
    
    def get_valid_actions(self, env, agent):
        """Obtiene acciones válidas"""
        try:
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
        except Exception as e:
            return [0]
    
    def evaluate_agents_comprehensive(self, agents, num_games=500):
        """Evaluación completa de los agentes"""
        print(f"🔍 Iniciando evaluación completa con {num_games} juegos...")
        
        # Métricas detalladas
        results = {
            'games_played': 0,
            'games_completed': 0,
            'wins': {"equipo_1": 0, "equipo_2": 0},
            'total_points': {"equipo_1": 0, "equipo_2": 0},
            'game_lengths': [],
            'phase_stats': {
                'MUS': {'mus_requested': 0, 'no_mus': 0},
                'GRANDE': {'envidos': 0, 'ordagos': 0, 'pasos': 0},
                'CHICA': {'envidos': 0, 'ordagos': 0, 'pasos': 0},
                'PARES': {'envidos': 0, 'ordagos': 0, 'pasos': 0},
                'JUEGO': {'envidos': 0, 'ordagos': 0, 'pasos': 0}
            },
            'decision_patterns': {agent: {'actions': [], 'phases': []} for agent in agents.keys()},
            'team_coordination': {'successful_supports': 0, 'total_opportunities': 0},
            'learning_indicators': {
                'strategic_decisions': 0,
                'random_decisions': 0,
                'optimal_decisions': 0
            }
        }
        
        for game in range(num_games):
            if game % 100 == 0:
                print(f"  Progreso: {game}/{num_games} juegos")
            
            env = mus.env()
            env.reset()
            results['games_played'] += 1
            
            game_length = 0
            states = {}
            
            # Inicializar estados
            for agent in env.agents:
                obs = env.observe(agent)
                states[agent] = self.process_observation(obs)
            
            while not all(env.dones.values()) and game_length < 1000:
                current_agent = env.agent_selection
                valid_actions = self.get_valid_actions(env, current_agent)
                
                # Configurar epsilon bajo para evaluación
                old_epsilon = agents[current_agent].epsilon
                agents[current_agent].epsilon = 0.01  # Muy bajo para evaluación
                
                action = agents[current_agent].act(states[current_agent], valid_actions)
                
                # Restaurar epsilon
                agents[current_agent].epsilon = old_epsilon
                
                # Registrar decisión
                results['decision_patterns'][current_agent]['actions'].append(action)
                results['decision_patterns'][current_agent]['phases'].append(env.fase_actual)
                
                # Analizar tipo de decisión
                if self._is_strategic_decision(env, current_agent, action):
                    results['learning_indicators']['strategic_decisions'] += 1
                elif self._is_optimal_decision(env, current_agent, action):
                    results['learning_indicators']['optimal_decisions'] += 1
                else:
                    results['learning_indicators']['random_decisions'] += 1
                
                # Estadísticas por fase
                if env.fase_actual in results['phase_stats']:
                    if action == 1:  # Envido
                        results['phase_stats'][env.fase_actual]['envidos'] += 1
                    elif action == 6:  # Órdago
                        results['phase_stats'][env.fase_actual]['ordagos'] += 1
                    elif action == 0:  # Paso
                        results['phase_stats'][env.fase_actual]['pasos'] += 1
                    elif env.fase_actual == "MUS":
                        if action == 2:
                            results['phase_stats']['MUS']['mus_requested'] += 1
                        elif action == 3:
                            results['phase_stats']['MUS']['no_mus'] += 1
                
                # Analizar coordinación de equipo
                if self._is_team_support_opportunity(env, current_agent, action):
                    results['team_coordination']['total_opportunities'] += 1
                    if self._is_successful_team_support(env, current_agent, action):
                        results['team_coordination']['successful_supports'] += 1
                
                env.step(action)
                game_length += 1
                
                # Actualizar estado
                if not env.dones.get(current_agent, False):
                    next_obs = env.observe(current_agent)
                    states[current_agent] = self.process_observation(next_obs)
            
            # Registrar resultado del juego
            results['game_lengths'].append(game_length)
            results['total_points']['equipo_1'] += env.puntos_equipos['equipo_1']
            results['total_points']['equipo_2'] += env.puntos_equipos['equipo_2']
            
            if env.puntos_equipos["equipo_1"] >= 30:
                results['wins']["equipo_1"] += 1
                results['games_completed'] += 1
            elif env.puntos_equipos["equipo_2"] >= 30:
                results['wins']["equipo_2"] += 1
                results['games_completed'] += 1
        
        # Calcular métricas finales
        self._calculate_final_metrics(results)
        
        return results
    
    def _is_strategic_decision(self, env, agent, action):
        """Determina si una decisión es estratégica"""
        # Decisiones estratégicas: envidos con buenas manos, órdagos calculados, etc.
        if env.fase_actual == "GRANDE" and action == 1:  # Envido en grande
            mano = env.manos[agent]
            figuras = sum(1 for carta in mano if carta[0] >= 10)
            return figuras >= 2
        elif env.fase_actual == "CHICA" and action == 1:  # Envido en chica
            mano = env.manos[agent]
            cartas_bajas = sum(1 for carta in mano if carta[0] <= 4)
            return cartas_bajas >= 2
        elif action == 6:  # Órdago
            return True  # Siempre consideramos órdago como estratégico
        return False
    
    def _is_optimal_decision(self, env, agent, action):
        """Determina si una decisión es óptima (heurística simple)"""
        if env.fase_actual == "MUS":
            mano = env.manos[agent]
            valores = [carta[0] for carta in mano]
            cartas_malas = sum(1 for v in valores if 5 <= v <= 7)
            optimal = 2 if cartas_malas >= 2 else 3
            return action == optimal
        return False
    
    def _is_team_support_opportunity(self, env, agent, action):
        """Identifica oportunidades de apoyo al equipo"""
        if env.equipo_apostador:
            equipo_actual = env.equipo_de_jugador[agent]
            return equipo_actual == env.equipo_apostador and env.apuesta_actual > 0
        return False
    
    def _is_successful_team_support(self, env, agent, action):
        """Determina si el apoyo al equipo fue exitoso"""
        return action in [1, 7]  # Envido o Quiero como apoyo
    
    def _calculate_final_metrics(self, results):
        """Calcula métricas finales"""
        total_games = results['games_completed']
        if total_games > 0:
            results['win_rate_eq1'] = results['wins']['equipo_1'] / total_games
            results['win_rate_eq2'] = results['wins']['equipo_2'] / total_games
            results['avg_game_length'] = np.mean(results['game_lengths'])
            results['avg_points_eq1'] = results['total_points']['equipo_1'] / results['games_played']
            results['avg_points_eq2'] = results['total_points']['equipo_2'] / results['games_played']
        
        # Métricas de aprendizaje
        total_decisions = sum(results['learning_indicators'].values())
        if total_decisions > 0:
            results['strategic_ratio'] = results['learning_indicators']['strategic_decisions'] / total_decisions
            results['optimal_ratio'] = results['learning_indicators']['optimal_decisions'] / total_decisions
            results['random_ratio'] = results['learning_indicators']['random_decisions'] / total_decisions
        
        # Coordinación de equipo
        if results['team_coordination']['total_opportunities'] > 0:
            results['team_coordination_rate'] = (
                results['team_coordination']['successful_supports'] / 
                results['team_coordination']['total_opportunities']
            )
        else:
            results['team_coordination_rate'] = 0
    
    def compare_with_random_agents(self, trained_agents, num_games=200):
        """Compara agentes entrenados con agentes aleatorios"""
        print(f"🎲 Comparando con agentes aleatorios ({num_games} juegos)...")
        
        # Crear agentes aleatorios
        random_agents = {}
        for i in range(4):
            team = "equipo_1" if i in [0, 2] else "equipo_2"
            random_agents[f"jugador_{i}"] = MARLAgent(21, 15, i, team)
            random_agents[f"jugador_{i}"].epsilon = 1.0  # Completamente aleatorio
        
        # Evaluar agentes entrenados
        trained_results = self.evaluate_agents_comprehensive(trained_agents, num_games)
        
        # Evaluar agentes aleatorios
        random_results = self.evaluate_agents_comprehensive(random_agents, num_games)
        
        # Comparar resultados
        comparison = {
            'trained': {
                'win_rate_eq1': trained_results['win_rate_eq1'],
                'win_rate_eq2': trained_results['win_rate_eq2'],
                'avg_game_length': trained_results['avg_game_length'],
                'strategic_ratio': trained_results['strategic_ratio'],
                'team_coordination_rate': trained_results['team_coordination_rate']
            },
            'random': {
                'win_rate_eq1': random_results['win_rate_eq1'],
                'win_rate_eq2': random_results['win_rate_eq2'],
                'avg_game_length': random_results['avg_game_length'],
                'strategic_ratio': random_results['strategic_ratio'],
                'team_coordination_rate': random_results['team_coordination_rate']
            }
        }
        
        # Calcular mejoras
        comparison['improvement'] = {}
        for metric in ['win_rate_eq1', 'win_rate_eq2', 'strategic_ratio', 'team_coordination_rate']:
            comparison['improvement'][metric] = (
                comparison['trained'][metric] - comparison['random'][metric]
            )
        
        # Mejora en duración (negativa es mejor)
        comparison['improvement']['avg_game_length'] = (
            comparison['random']['avg_game_length'] - comparison['trained']['avg_game_length']
        )
        
        return comparison
    
    def generate_detailed_report(self, results, comparison=None):
        """Genera un reporte detallado de evaluación"""
        report = []
        
        report.append("=" * 80)
        report.append("REPORTE DETALLADO DE EVALUACIÓN DE AGENTES")
        report.append("=" * 80)
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Juegos evaluados: {results['games_played']}")
        report.append(f"Juegos completados: {results['games_completed']}")
        
        report.append("\n" + "─" * 50)
        report.append("RENDIMIENTO GENERAL")
        report.append("─" * 50)
        report.append(f"Tasa de victoria Equipo 1: {results['win_rate_eq1']:.2%}")
        report.append(f"Tasa de victoria Equipo 2: {results['win_rate_eq2']:.2%}")
        report.append(f"Duración promedio de juego: {results['avg_game_length']:.1f} turnos")
        report.append(f"Puntos promedio Equipo 1: {results['avg_points_eq1']:.1f}")
        report.append(f"Puntos promedio Equipo 2: {results['avg_points_eq2']:.1f}")
        
        report.append("\n" + "─" * 50)
        report.append("INDICADORES DE APRENDIZAJE")
        report.append("─" * 50)
        report.append(f"Decisiones estratégicas: {results['strategic_ratio']:.2%}")
        report.append(f"Decisiones óptimas: {results['optimal_ratio']:.2%}")
        report.append(f"Decisiones aleatorias: {results['random_ratio']:.2%}")
        report.append(f"Coordinación de equipo: {results['team_coordination_rate']:.2%}")
        
        report.append("\n" + "─" * 50)
        report.append("ESTADÍSTICAS POR FASE")
        report.append("─" * 50)
        for fase, stats in results['phase_stats'].items():
            report.append(f"\n{fase}:")
            for action, count in stats.items():
                report.append(f"  {action}: {count}")
        
        if comparison:
            report.append("\n" + "─" * 50)
            report.append("COMPARACIÓN CON AGENTES ALEATORIOS")
            report.append("─" * 50)
            report.append("Mejoras respecto a agentes aleatorios:")
            for metric, improvement in comparison['improvement'].items():
                symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                report.append(f"  {symbol} {metric}: {improvement:+.2%}")
        
        report.append("\n" + "─" * 50)
        report.append("CONCLUSIONES")
        report.append("─" * 50)
        
        # Análisis automático
        if results['strategic_ratio'] > 0.3:
            report.append("✅ Los agentes muestran comportamiento estratégico significativo")
        else:
            report.append("❌ Los agentes necesitan más entrenamiento estratégico")
        
        if results['team_coordination_rate'] > 0.5:
            report.append("✅ Buena coordinación entre compañeros de equipo")
        else:
            report.append("⚠️ La coordinación de equipo necesita mejoras")
        
        if comparison and comparison['improvement']['strategic_ratio'] > 0.2:
            report.append("✅ Mejora significativa respecto a agentes aleatorios")
        elif comparison:
            report.append("⚠️ Mejora limitada respecto a agentes aleatorios")
        
        return "\n".join(report)
    
    def create_evaluation_dashboard(self, results, comparison=None, save_path=None):
        """Crea un dashboard visual de evaluación"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Tasas de victoria
        ax1 = plt.subplot(3, 4, 1)
        teams = ['Equipo 1', 'Equipo 2']
        win_rates = [results['win_rate_eq1'], results['win_rate_eq2']]
        colors = ['blue', 'red']
        bars = ax1.bar(teams, win_rates, color=colors, alpha=0.7)
        ax1.set_title('Tasas de Victoria')
        ax1.set_ylabel('Tasa de Victoria')
        ax1.set_ylim(0, 1)
        for bar, rate in zip(bars, win_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 2. Distribución de duración de juegos
        ax2 = plt.subplot(3, 4, 2)
        ax2.hist(results['game_lengths'], bins=30, alpha=0.7, color='green')
        ax2.set_title('Distribución Duración de Juegos')
        ax2.set_xlabel('Turnos')
        ax2.set_ylabel('Frecuencia')
        ax2.axvline(results['avg_game_length'], color='red', linestyle='--', 
                   label=f'Promedio: {results["avg_game_length"]:.1f}')
        ax2.legend()
        
        # 3. Indicadores de aprendizaje
        ax3 = plt.subplot(3, 4, 3)
        learning_metrics = ['Estratégicas', 'Óptimas', 'Aleatorias']
        learning_values = [results['strategic_ratio'], results['optimal_ratio'], results['random_ratio']]
        colors_learning = ['gold', 'green', 'gray']
        pie = ax3.pie(learning_values, labels=learning_metrics, colors=colors_learning, autopct='%1.1f%%')
        ax3.set_title('Tipos de Decisiones')
        
        # 4. Coordinación de equipo
        ax4 = plt.subplot(3, 4, 4)
        coord_data = ['Exitosa', 'Fallida']
        coord_values = [results['team_coordination_rate'], 1 - results['team_coordination_rate']]
        ax4.pie(coord_values, labels=coord_data, colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        ax4.set_title('Coordinación de Equipo')
        
        # 5-8. Estadísticas por fase
        phases = ['GRANDE', 'CHICA', 'PARES', 'JUEGO']
        for i, phase in enumerate(phases):
            ax = plt.subplot(3, 4, 5 + i)
            if phase in results['phase_stats']:
                stats = results['phase_stats'][phase]
                if phase == 'MUS':
                    actions = list(stats.keys())
                    counts = list(stats.values())
                else:
                    actions = ['Envidos', 'Órdagos', 'Pasos']
                    counts = [stats.get('envidos', 0), stats.get('ordagos', 0), stats.get('pasos', 0)]
                
                ax.bar(actions, counts, alpha=0.7)
                ax.set_title(f'Acciones en {phase}')
                ax.tick_params(axis='x', rotation=45)
        
        # 9. Comparación con aleatorios (si disponible)
        if comparison:
            ax9 = plt.subplot(3, 4, 9)
            metrics = list(comparison['improvement'].keys())
            improvements = list(comparison['improvement'].values())
            colors_comp = ['green' if x > 0 else 'red' for x in improvements]
            
            bars = ax9.bar(range(len(metrics)), improvements, color=colors_comp, alpha=0.7)
            ax9.set_title('Mejora vs Agentes Aleatorios')
            ax9.set_xticks(range(len(metrics)))
            ax9.set_xticklabels(metrics, rotation=45, ha='right')
            ax9.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax9.set_ylabel('Mejora')
        
        # 10. Puntos promedio por equipo
        ax10 = plt.subplot(3, 4, 10)
        teams = ['Equipo 1', 'Equipo 2']
        avg_points = [results['avg_points_eq1'], results['avg_points_eq2']]
        ax10.bar(teams, avg_points, color=['blue', 'red'], alpha=0.7)
        ax10.set_title('Puntos Promedio por Equipo')
        ax10.set_ylabel('Puntos')
        
        # 11. Evolución temporal (si hay datos históricos)
        ax11 = plt.subplot(3, 4, 11)
        ax11.text(0.5, 0.5, 'Datos de evolución\ntemporal no disponibles', 
                 ha='center', va='center', transform=ax11.transAxes)
        ax11.set_title('Evolución Temporal')
        
        # 12. Resumen de métricas clave
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        summary_text = f"""
RESUMEN EJECUTIVO

🎯 Rendimiento General:
   Win Rate: {(results['win_rate_eq1'] + results['win_rate_eq2'])/2:.1%}
   
🧠 Aprendizaje:
   Estratégico: {results['strategic_ratio']:.1%}
   
🤝 Coordinación:
   Equipo: {results['team_coordination_rate']:.1%}
   
⏱️ Eficiencia:
   Duración: {results['avg_game_length']:.0f} turnos
        """
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Dashboard guardado en: {save_path}")
        
        plt.show()
    
    def save_evaluation_results(self, results, comparison=None, filename=None):
        """Guarda los resultados de evaluación"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': results,
            'comparison_with_random': comparison
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Resultados guardados en: {filename}")
        return filename

def main():
    """Función principal para evaluación interactiva"""
    print("🔍 EVALUADOR DE AGENTES MARL - MUS")
    print("=" * 50)
    
    evaluator = AgentEvaluator()
    
    # Cargar agentes
    print("📂 Cargando agentes entrenados...")
    agents = evaluator.load_agents()
    
    # Configuración de evaluación
    num_games = int(input("Número de juegos para evaluación (default 500): ") or "500")
    
    # Evaluación principal
    print(f"\n🎮 Iniciando evaluación con {num_games} juegos...")
    results = evaluator.evaluate_agents_comprehensive(agents, num_games)
    
    # Comparación con aleatorios
    compare_random = input("\n¿Comparar con agentes aleatorios? (y/n): ").strip().lower() == 'y'
    comparison = None
    if compare_random:
        comparison = evaluator.compare_with_random_agents(agents, num_games // 2)
    
    # Generar reporte
    print("\n📋 Generando reporte...")
    report = evaluator.generate_detailed_report(results, comparison)
    print(report)
    
    # Guardar resultados
    save_results = input("\n💾 ¿Guardar resultados? (y/n): ").strip().lower() == 'y'
    if save_results:
        evaluator.save_evaluation_results(results, comparison)
    
    # Crear dashboard
    create_dashboard = input("\n📊 ¿Crear dashboard visual? (y/n): ").strip().lower() == 'y'
    if create_dashboard:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_path = f"evaluation_dashboard_{timestamp}.png"
        evaluator.create_evaluation_dashboard(results, comparison, dashboard_path)
    
    print("\n✅ Evaluación completada!")

if __name__ == "__main__":
    main()