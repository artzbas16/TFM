import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class MARLAgent:
    def __init__(self, state_size, action_size, agent_id, team):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.team = team
        
        # Hiperparámetros
        self.gamma = 0.95    # Factor de descuento
        self.epsilon = 1.0   # Tasa de exploración
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        self.steps = 0  # Contador de pasos para actualizar target network
        
        # Red neuronal para Q-learning
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Optimizador
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
        return model
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, valid_actions):
        if len(valid_actions) == 0:
            return 0  # Acción por defecto si no hay acciones válidas
            
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state_tensor).squeeze()
        
        # Filtrar solo acciones válidas
        valid_act_values = -np.inf * np.ones(self.action_size)
        for action in valid_actions:
            if action < len(act_values):
                valid_act_values[action] = act_values[action].item()
        
        return np.argmax(valid_act_values)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.steps += 1
        
        # Actualización periódica del target network
        if self.steps % 100 == 0:
            self.update_target_model()
            
    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filename)
        
    def load(self, filename):
        try:
            checkpoint = torch.load(filename, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps = checkpoint.get('steps', 0)
            print(f"Modelo cargado exitosamente desde {filename}")
        except FileNotFoundError:
            print(f"No se encontró el archivo {filename}, usando modelo nuevo")
        except Exception as e:
            print(f"Error cargando modelo: {e}")

    def get_reward_for_action(self, fase, action, won_round, team_points, opponent_points):
        """Calcula recompensas específicas según la fase y resultado de acción"""
        base_reward = 0
        
        # Recompensas por ganar/perder la ronda
        if won_round:
            base_reward += 10
        elif won_round is False:  # Perdió explícitamente
            base_reward -= 5
            
        # Recompensas por diferencia de puntos
        point_diff = team_points - opponent_points
        base_reward += point_diff * 0.5
        
        # Recompensas específicas por acción y fase
        if fase == "MUS":
            if action == 2:  # Pedir Mus
                base_reward += 1  # Recompensa por ser activo
            elif action == 3:  # No Mus
                base_reward += 0.5  # Recompensa menor por continuar
                
        elif fase in ['GRANDE', 'CHICA', 'PARES', 'JUEGO']:
            if action == 1:  # Envido
                base_reward += 2 if won_round else -1
            elif action == 6:  # Órdago
                base_reward += 5 if won_round else -3
            elif action == 7:  # Quiero (aceptar)
                base_reward += 3 if won_round else -2
            elif action == 5:  # No quiero (rechazar)
                base_reward += 1  # Evitar riesgo
            elif action == 0:  # Paso
                base_reward += 0.5  # Neutro
                
        return base_reward

    def share_experience(self, other_agent):
        """Compartir experiencia con otro agente del mismo equipo"""
        if len(self.memory) > 10:
            shared_memories = random.sample(list(self.memory), min(10, len(self.memory)//4))
            for memory in shared_memories:
                other_agent.memory.append(memory)