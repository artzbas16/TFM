import numpy as np
from mus_env import mus
from marl_agent import MARLAgent
import torch
import random
import time
from training_controller import TrainingController, LearningEvaluator

def process_observation(obs):
    """Procesa la observación del entorno para el agente MARL"""
    try:
        cartas_flat = obs["cartas"].flatten()
        fase_onehot = np.zeros(7)
        fase_onehot[obs["fase"]] = 1
        turno_onehot = np.zeros(4)
        turno_onehot[obs["turno"]] = 1
        
        apuesta_norm = obs.get("apuesta_actual", 0) / 30.0
        equipo_apostador = obs.get("equipo_apostador", 0) / 2.0
        
        state = np.concatenate([
            cartas_flat,
            fase_onehot,
            turno_onehot,
            [apuesta_norm],
            [equipo_apostador]
        ])
        
        return state
    except Exception as e:
        print(f"Error procesando observación: {e}")
        return np.zeros(21)

def get_valid_actions(env, agent):
    """Obtiene las acciones válidas para un agente en el estado actual"""
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
                equipo_apostador = env.equipo_apostador
                
                if equipo_actual != equipo_apostador:
                    valid.extend([5, 7])
                    
            if not hasattr(env, 'hay_ordago') or not env.hay_ordago:
                valid.append(6)
                
            return valid
        else:
            return [0]
    except Exception as e:
        print(f"Error obteniendo acciones válidas: {e}")
        return [0]

def calculate_detailed_rewards(env, agent, action, prev_points):
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
        if action == 2:
            reward += 0.5
        elif action == 3:
            reward += 1
            
    elif env.fase_actual == "DESCARTE":
        reward += 0.5
        
    elif env.fase_actual in ['GRANDE', 'CHICA', 'PARES', 'JUEGO']:
        if action == 1:  # Envido
            if point_change > 0:
                reward += 3
            elif point_change < 0:
                reward -= 2
            else:
                reward += 0.5
                
        elif action == 6:  # Órdago
            if point_change > 0:
                reward += 5
            elif point_change < 0:
                reward -= 4
            else:
                reward += 1
                
        elif action == 7:  # Quiero
            if point_change > 0:
                reward += 2
            elif point_change < 0:
                reward -= 1
                
        elif action == 5:  # No quiero
            reward += 0.5
            
        elif action == 0:  # Paso
            reward += 0.2
    
    # Recompensa por diferencia de puntos
    point_diff = current_points - opponent_points
    reward += point_diff * 0.1
    
    # Recompensa por ganar la partida
    if current_points >= 30:
        reward += 15
    elif opponent_points >= 30:
        reward -= 10
    
    return reward

def main():
    print("=== ENTRENAMIENTO MARL CONTROLADO PARA MUS ===")
    
    # Inicializar controladores
    controller = TrainingController()
    evaluator = LearningEvaluator()
    
    # Configuración interactiva
    print("\nConfiguración de entrenamiento:")
    
    # Objetivos de entrenamiento
    duration_input = input("Duración máxima en minutos (Enter para 60): ").strip()
    duration = int(duration_input) if duration_input else 60
    
    episodes_input = input("Número máximo de episodios (Enter para 1000): ").strip()
    max_episodes = int(episodes_input) if episodes_input else 1000
    
    target_win_rate_input = input("Tasa de victoria objetivo 0-1 (Enter para 0.6): ").strip()
    target_win_rate = float(target_win_rate_input) if target_win_rate_input else 0.6
    
    controller.set_training_target(
        duration_minutes=duration,
        episodes=max_episodes,
        target_win_rate=target_win_rate
    )
    
    # Configuración de evaluación
    eval_interval_input = input("Intervalo de evaluación en episodios (Enter para 100): ").strip()
    eval_interval = int(eval_interval_input) if eval_interval_input else 100
    
    save_interval_input = input("Intervalo de guardado en episodios (Enter para 50): ").strip()
    save_interval = int(save_interval_input) if save_interval_input else 50
    
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
        try:
            agents[f"jugador_{i}"].load(model_path)
            print(f"Modelo cargado para jugador_{i}")
        except:
            print(f"Usando modelo nuevo para jugador_{i}")

    # Establecer línea base
    print("\nEstableciendo línea base de rendimiento...")
    baseline_result = evaluator.evaluate_learning_progress(agents, 50)
    evaluator.set_baseline(
        baseline_result['win_rate_eq1'],
        baseline_result['win_rate_eq2'],
        baseline_result['avg_game_length']
    )
    
    # Variables de seguimiento
    episode_rewards = {agent: [] for agent in env.agents}
    total_wins = {"equipo_1": 0, "equipo_2": 0}
    best_win_rate = 0.0
    
    # Iniciar entrenamiento
    controller.start_training()
    episode = 0
    
    print(f"\n🚀 Iniciando entrenamiento...")
    print(f"Presiona Ctrl+C para pausar/reanudar o detener")
    
    try:
        while controller.should_continue_training() and episode < max_episodes:
            # Verificar si está pausado
            while controller.pause_training and controller.should_continue_training():
                print("⏸️ Entrenamiento pausado. Presiona Enter para reanudar o 'q' para detener...")
                user_input = input().strip().lower()
                if user_input == 'q':
                    controller.stop_training_session()
                    break
                else:
                    controller.resume_training()
            
            if not controller.should_continue_training():
                break
            
            # Ejecutar episodio
            env.reset()
            all_rewards = {agent: 0 for agent in env.agents}
            prev_points = {agent: 0 for agent in env.agents}
            states = {}
            
            # Inicializar estados
            for agent in env.agents:
                obs = env.observe(agent)
                states[agent] = process_observation(obs)

            step_count = 0
            max_steps = 1000

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
                    reward = calculate_detailed_rewards(env, current_agent, action, prev_points)
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
                    
                    # Entrenar
                    if len(agents[current_agent].memory) > 64:
                        agents[current_agent].replay()
                    
                    # Compartir experiencia ocasionalmente
                    if random.random() < 0.05:
                        team = agents[current_agent].team
                        teammates = [name for name, agent in agents.items() 
                                   if agent.team == team and name != current_agent]
                        if teammates:
                            teammate = random.choice(teammates)
                            agents[current_agent].share_experience(agents[teammate])
                            
                except Exception as e:
                    print(f"Error en episodio {episode}: {e}")
                    break
            
            # Registrar resultados del episodio
            for agent in env.agents:
                episode_rewards[agent].append(all_rewards[agent])
            
            if env.puntos_equipos["equipo_1"] >= 30:
                total_wins["equipo_1"] += 1
            elif env.puntos_equipos["equipo_2"] >= 30:
                total_wins["equipo_2"] += 1
            
            # Actualizar controlador
            controller.update_episode(episode)
            
            # Añadir estadísticas
            avg_reward = np.mean([all_rewards[agent] for agent in env.agents])
            controller.add_training_stats({
                'episode': episode,
                'avg_reward': avg_reward,
                'total_wins_eq1': total_wins["equipo_1"],
                'total_wins_eq2': total_wins["equipo_2"],
                'avg_epsilon': np.mean([agent.epsilon for agent in agents.values()])
            })
            
            # Evaluación periódica
            if episode % eval_interval == 0 and episode > 0:
                print(f"\n📊 Evaluando en episodio {episode}...")
                eval_result = evaluator.evaluate_learning_progress(agents, 100)
                controller.add_evaluation_result(eval_result)
                
                # Mostrar progreso
                progress = controller.get_training_progress()
                print(f"⏱️ Progreso: {progress['progress_percentage']:.1f}% - "
                      f"Tiempo: {progress['elapsed_time']/60:.1f}min")
                
                # Verificar si ha aprendido
                if evaluator.has_learned_significantly():
                    print("🎉 ¡Los agentes han aprendido significativamente!")
                    
                    # Preguntar si continuar
                    continue_training = input("¿Continuar entrenamiento? (y/n): ").strip().lower()
                    if continue_training != 'y':
                        controller.stop_training_session()
                        break
                
                # Guardar mejor modelo
                avg_win_rate = (eval_result['win_rate_eq1'] + eval_result['win_rate_eq2']) / 2
                if avg_win_rate > best_win_rate:
                    best_win_rate = avg_win_rate
                    for i, agent_name in enumerate(agents.keys()):
                        agents[agent_name].save(f"best_model_jugador_{i}.pth")
                    print(f"💾 Mejores modelos guardados (tasa: {avg_win_rate:.2%})")
            
            # Guardado periódico
            if episode % save_interval == 0 and episode > 0:
                for i, agent_name in enumerate(agents.keys()):
                    agents[agent_name].save(f"model_jugador_{i}_ep_{episode}.pth")
                
                # Mostrar progreso básico
                total_games = total_wins["equipo_1"] + total_wins["equipo_2"]
                current_win_rate_eq1 = total_wins["equipo_1"] / total_games if total_games > 0 else 0.5
                
                print(f"📈 Episodio {episode}: WinRate={current_win_rate_eq1:.2%}, "
                      f"Epsilon={np.mean([agent.epsilon for agent in agents.values()]):.3f}")
            
            episode += 1
    
    except KeyboardInterrupt:
        print("\n⏸️ Entrenamiento interrumpido por el usuario")
        choice = input("¿Pausar (p), Detener (s) o Continuar (c)? ").strip().lower()
        if choice == 'p':
            controller.pause_training_session()
        elif choice == 's':
            controller.stop_training_session()
        # Si es 'c' o cualquier otra cosa, continúa
    
    # Finalizar entrenamiento
    controller.stop_training_session()
    
    # Guardar modelos finales
    print("\n💾 Guardando modelos finales...")
    for i, agent_name in enumerate(agents.keys()):
        agents[agent_name].save(f"model_jugador_{i}_final.pth")
    
    # Evaluación final
    print("\n🔍 Evaluación final...")
    final_eval = evaluator.evaluate_learning_progress(agents, 200)
    
    # Generar reporte
    print("\n" + "="*60)
    print(evaluator.generate_learning_report())
    print("="*60)
    
    # Guardar sesión
    session_file = controller.save_training_session()
    
    # Generar gráficos
    try:
        evaluator.plot_learning_curves(f"learning_curves_{episode}.png")
    except Exception as e:
        print(f"Error generando gráficos: {e}")
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"📁 Sesión guardada en: {session_file}")
    print(f"🏆 Mejor tasa de victoria: {best_win_rate:.2%}")
    
    if evaluator.has_learned_significantly():
        print("🎉 ¡Los agentes HAN APRENDIDO exitosamente!")
    else:
        print("⚠️ Los agentes necesitan más entrenamiento")

if __name__ == "__main__":
    main()