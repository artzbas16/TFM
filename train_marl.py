import numpy as np
from mus_env import mus
from marl_agent import MARLAgent
import torch
import random
import pickle
import os
from datetime import datetime

# Hiperparámetros
EPISODES = 5000
BATCH_SIZE = 64
SAVE_INTERVAL = 100
EVALUATION_INTERVAL = 250
TARGET_WIN_RATE = 0.55  # Objetivo de tasa de victoria para considerar entrenamiento exitoso

def process_observation(obs):
    """Procesa la observación del entorno para el agente MARL"""
    try:
        # Extraer información básica
        cartas_flat = obs["cartas"].flatten()
        fase_onehot = np.zeros(7)
        fase_onehot[obs["fase"]] = 1
        turno_onehot = np.zeros(4)
        turno_onehot[obs["turno"]] = 1
        
        # Información adicional del juego
        apuesta_norm = obs.get("apuesta_actual", 0) / 30.0
        equipo_apostador = obs.get("equipo_apostador", 0) / 2.0
        
        # Concatenar todos los features
        state = np.concatenate([
            cartas_flat,           # 8 valores
            fase_onehot,          # 7 valores  
            turno_onehot,         # 4 valores
            [apuesta_norm],       # 1 valor
            [equipo_apostador]    # 1 valor
        ])
        
        return state
    except Exception as e:
        print(f"Error procesando observación: {e}")
        return np.zeros(21)

def get_valid_actions(env, agent):
    """Obtiene las acciones válidas para un agente en el estado actual"""
    try:
        if env.fase_actual == "MUS":
            return [2, 3]  # Mus o No Mus
        elif env.fase_actual == "DESCARTE":
            return [4] + list(range(11, 15))  # OK + selección cartas
        elif env.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
            if agent not in env.jugadores_que_pueden_hablar:
                return [0]  # Solo puede pasar
            
            valid = [0, 1]  # Paso, Envido
            
            if env.apuesta_actual > 0:
                equipo_actual = env.equipo_de_jugador[agent]
                equipo_apostador = env.equipo_apostador
                
                if equipo_actual != equipo_apostador:
                    valid.extend([5, 7])  # No quiero, Quiero
                    
            if not hasattr(env, 'hay_ordago') or not env.hay_ordago:
                valid.append(6)  # Órdago
                
            return valid
        else:
            return [0]  # Solo paso por defecto
    except Exception as e:
        print(f"Error obteniendo acciones válidas: {e}")
        return [0]

def calculate_detailed_rewards(env, agent, action, prev_points, won_round=None):
    """Calcula recompensas detalladas basadas en el estado del juego"""
    reward = 0
    equipo = env.equipo_de_jugador[agent]
    current_points = env.puntos_equipos[equipo]
    opponent_points = env.puntos_equipos["equipo_2" if equipo == "equipo_1" else "equipo_1"]
    
    # Recompensa por cambio en puntos
    point_change = current_points - prev_points.get(agent, 0)
    reward += point_change * 2
    
    # Recompensas específicas por fase y acción
    if env.fase_actual == "MUS":
        if action == 2:  # Pedir Mus
            # Recompensa por ser activo, pero penalización si no mejora
            reward += 0.5
        elif action == 3:  # No Mus
            reward += 1  # Recompensa por continuar el juego
            
    elif env.fase_actual == "DESCARTE":
        reward += 0.5  # Recompensa por participar
        
    elif env.fase_actual in ['GRANDE', 'CHICA', 'PARES', 'JUEGO']:
        if action == 1:  # Envido
            if point_change > 0:
                reward += 3  # Buena apuesta
            elif point_change < 0:
                reward -= 2  # Mala apuesta
            else:
                reward += 0.5  # Neutral
                
        elif action == 6:  # Órdago
            if point_change > 0:
                reward += 5  # Órdago exitoso
            elif point_change < 0:
                reward -= 4  # Órdago fallido
            else:
                reward += 1  # Órdago neutro
                
        elif action == 7:  # Quiero (aceptar)
            if point_change > 0:
                reward += 2  # Buena aceptación
            elif point_change < 0:
                reward -= 1  # Mala aceptación
                
        elif action == 5:  # No quiero (rechazar)
            reward += 0.5  # Evitar riesgo puede ser bueno
            
        elif action == 0:  # Paso
            reward += 0.2  # Neutral
    
    # Recompensa por diferencia de puntos
    point_diff = current_points - opponent_points
    reward += point_diff * 0.1
    
    # Recompensa por ganar la partida
    if current_points >= 30:
        reward += 15
    elif opponent_points >= 30:
        reward -= 10
    
    # Penalización por juegos muy largos (evitar loops infinitos)
    if env.fase_actual == "RECUENTO":
        if current_points < 5:  # Juego muy defensivo
            reward -= 2
            
    return reward

def evaluate_agents(agents, episodes=50):
    """Evalúa el rendimiento de los agentes entrenados"""
    wins = {"equipo_1": 0, "equipo_2": 0}
    
    for _ in range(episodes):
        env = mus.env()
        env.reset()
        
        states = {}
        for agent in env.agents:
            obs = env.observe(agent)
            states[agent] = process_observation(obs)
        
        while not all(env.dones.values()):
            try:
                current_agent = env.agent_selection
                valid_actions = get_valid_actions(env, current_agent)
                
                # Usar epsilon bajo para evaluación (más explotación)
                old_epsilon = agents[current_agent].epsilon
                agents[current_agent].epsilon = 0.05
                
                action = agents[current_agent].act(states[current_agent], valid_actions)
                
                # Restaurar epsilon
                agents[current_agent].epsilon = old_epsilon
                
                env.step(action)
                
                if not env.dones.get(current_agent, False):
                    next_obs = env.observe(current_agent)
                    states[current_agent] = process_observation(next_obs)
                    
            except Exception as e:
                print(f"Error en evaluación: {e}")
                break
        
        # Contar victorias
        if env.puntos_equipos["equipo_1"] >= 30:
            wins["equipo_1"] += 1
        elif env.puntos_equipos["equipo_2"] >= 30:
            wins["equipo_2"] += 1
    
    total_games = wins["equipo_1"] + wins["equipo_2"]
    if total_games > 0:
        win_rate_eq1 = wins["equipo_1"] / total_games
        win_rate_eq2 = wins["equipo_2"] / total_games
    else:
        win_rate_eq1 = win_rate_eq2 = 0.5
    
    return win_rate_eq1, win_rate_eq2, wins

def save_training_stats(stats, filename="training_stats.pkl"):
    """Guarda estadísticas de entrenamiento"""
    with open(filename, 'wb') as f:
        pickle.dump(stats, f)

def load_training_stats(filename="training_stats.pkl"):
    """Carga estadísticas de entrenamiento"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

def main():
    print("Iniciando entrenamiento MARL para Mus...")
    print(f"Episodios: {EPISODES}")
    print(f"Intervalo de guardado: {SAVE_INTERVAL}")
    print(f"Intervalo de evaluación: {EVALUATION_INTERVAL}")
    
    # Inicializar entorno y agentes
    env = mus.env()
    state_size = 21
    action_size = 15
    agents = {}

    for i in range(4):
        team = "equipo_1" if i in [0, 2] else "equipo_2"
        agents[f"jugador_{i}"] = MARLAgent(state_size, action_size, i, team)
        
        # Intentar cargar modelo existente
        model_path = f"model_jugador_{i}.pth"
        if os.path.exists(model_path):
            agents[f"jugador_{i}"].load(model_path)
            print(f"Modelo cargado para jugador_{i}")

    # Cargar estadísticas previas
    training_stats = load_training_stats()
    
    # Variables de seguimiento
    episode_rewards = {agent: [] for agent in env.agents}
    total_wins = {"equipo_1": 0, "equipo_2": 0}
    best_win_rate = 0.0

    # Bucle de entrenamiento
    for episode in range(EPISODES):
        env.reset()
        all_rewards = {agent: 0 for agent in env.agents}
        prev_points = {agent: 0 for agent in env.agents}
        states = {}
        episode_experiences = []
        
        # Inicializar estados
        for agent in env.agents:
            obs = env.observe(agent)
            states[agent] = process_observation(obs)

        step_count = 0
        max_steps = 1000  # Evitar episodios infinitos

        while not all(env.dones.values()) and step_count < max_steps:
            try:
                current_agent = env.agent_selection
                valid_actions = get_valid_actions(env, current_agent)
                
                # Tomar acción
                action = agents[current_agent].act(states[current_agent], valid_actions)
                
                # Guardar estado previo
                prev_state = states[current_agent].copy()
                prev_points_agent = env.puntos_equipos[env.equipo_de_jugador[current_agent]]
                
                # Ejecutar acción
                env.step(action)
                step_count += 1
                
                # Calcular recompensa
                reward = calculate_detailed_rewards(
                    env, current_agent, action, prev_points
                )
                all_rewards[current_agent] += reward
                
                # Obtener nuevo estado
                if not env.dones.get(current_agent, False):
                    next_obs = env.observe(current_agent)
                    next_state = process_observation(next_obs)
                else:
                    next_state = np.zeros_like(states[current_agent])
                
                # Guardar experiencia
                done = env.dones.get(current_agent, False)
                agents[current_agent].remember(
                    prev_state, action, reward, next_state, done
                )
                
                # Actualizar estado
                states[current_agent] = next_state
                prev_points[current_agent] = env.puntos_equipos[env.equipo_de_jugador[current_agent]]
                
                # Entrenar periódicamente
                if len(agents[current_agent].memory) > BATCH_SIZE:
                    agents[current_agent].replay()
                
                # Compartir experiencia entre compañeros de equipo ocasionalmente
                if random.random() < 0.05:  # 5% de probabilidad
                    team = agents[current_agent].team
                    teammates = [name for name, agent in agents.items() if agent.team == team and name != current_agent]
                    if teammates:
                        teammate = random.choice(teammates)
                        agents[current_agent].share_experience(agents[teammate])
                        
            except Exception as e:
                print(f"Error en episodio {episode}, paso {step_count}: {e}")
                break
        
        # Registrar recompensas del episodio
        for agent in env.agents:
            episode_rewards[agent].append(all_rewards[agent])
        
        # Contar victorias
        if env.puntos_equipos["equipo_1"] >= 30:
            total_wins["equipo_1"] += 1
        elif env.puntos_equipos["equipo_2"] >= 30:
            total_wins["equipo_2"] += 1

        # Evaluación periódica
        if episode % EVALUATION_INTERVAL == 0 and episode > 0:
            print(f"\n--- Evaluación en episodio {episode} ---")
            win_rate_eq1, win_rate_eq2, eval_wins = evaluate_agents(agents, 100)
            
            print(f"Tasa de victoria Equipo 1: {win_rate_eq1:.3f}")
            print(f"Tasa de victoria Equipo 2: {win_rate_eq2:.3f}")
            print(f"Victorias evaluación - Eq1: {eval_wins['equipo_1']}, Eq2: {eval_wins['equipo_2']}")
            
            # Guardar mejor modelo
            avg_win_rate = (win_rate_eq1 + win_rate_eq2) / 2
            if avg_win_rate > best_win_rate:
                best_win_rate = avg_win_rate
                for i, agent_name in enumerate(agents.keys()):
                    agents[agent_name].save(f"best_model_jugador_{i}.pth")
                print(f"Nuevos mejores modelos guardados (tasa: {avg_win_rate:.3f})")
            
            # Guardar estadísticas
            stats_entry = {
                'episode': episode,
                'win_rate_eq1': win_rate_eq1,
                'win_rate_eq2': win_rate_eq2,
                'avg_reward_eq1': np.mean([episode_rewards[f"jugador_{i}"][-EVALUATION_INTERVAL:] for i in [0, 2]]),
                'avg_reward_eq2': np.mean([episode_rewards[f"jugador_{i}"][-EVALUATION_INTERVAL:] for i in [1, 3]]),
                'total_wins': total_wins.copy()
            }
            training_stats.append(stats_entry)
            save_training_stats(training_stats)

        # Guardar modelos periódicamente
        if episode % SAVE_INTERVAL == 0 and episode > 0:
            for i, agent_name in enumerate(agents.keys()):
                agents[agent_name].save(f"model_jugador_{i}_ep_{episode}.pth")
            
            # Mostrar progreso
            avg_reward = np.mean([np.mean(episode_rewards[agent][-SAVE_INTERVAL:]) for agent in env.agents])
            total_games = total_wins["equipo_1"] + total_wins["equipo_2"]
            if total_games > 0:
                current_win_rate_eq1 = total_wins["equipo_1"] / total_games
            else:
                current_win_rate_eq1 = 0.5
                
            print(f"Episodio {episode}/{EPISODES}")
            print(f"Recompensa promedio: {avg_reward:.2f}")
            print(f"Victorias totales - Eq1: {total_wins['equipo_1']}, Eq2: {total_wins['equipo_2']}")
            print(f"Tasa de victoria Eq1: {current_win_rate_eq1:.3f}")
            print(f"Epsilon promedio: {np.mean([agent.epsilon for agent in agents.values()]):.3f}")
            print("-" * 50)

    # Guardar modelos finales
    print("\nEntrenamiento completado. Guardando modelos finales...")
    for i, agent_name in enumerate(agents.keys()):
        agents[agent_name].save(f"model_jugador_{i}_final.pth")
    
    # Evaluación final
    print("\nEvaluación final...")
    final_win_rate_eq1, final_win_rate_eq2, final_wins = evaluate_agents(agents, 200)
    
    print(f"\n=== RESULTADOS FINALES ===")
    print(f"Episodios entrenados: {EPISODES}")
    print(f"Tasa de victoria final Equipo 1: {final_win_rate_eq1:.3f}")
    print(f"Tasa de victoria final Equipo 2: {final_win_rate_eq2:.3f}")
    print(f"Mejor tasa de victoria alcanzada: {best_win_rate:.3f}")
    print(f"Victorias evaluación final - Eq1: {final_wins['equipo_1']}, Eq2: {final_wins['equipo_2']}")
    
    # Guardar estadísticas finales
    final_stats = {
        'final_episode': EPISODES,
        'final_win_rate_eq1': final_win_rate_eq1,
        'final_win_rate_eq2': final_win_rate_eq2,
        'best_win_rate': best_win_rate,
        'total_training_wins': total_wins,
        'final_evaluation_wins': final_wins,
        'training_date': datetime.now().isoformat()
    }
    
    training_stats.append(final_stats)
    save_training_stats(training_stats)
    
    print(f"\nEstadísticas guardadas en training_stats.pkl")
    print("Entrenamiento completado exitosamente!")

if __name__ == "__main__":
    main()