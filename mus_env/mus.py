from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces
import numpy as np
import random

class MusEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "mus_v0"}

    def __init__(self):
        super().__init__()
        self.agents = [f"jugador_{i}" for i in range(4)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}

        self.fases = ["MUS", "DESCARTE", "GRANDE", "CHICA", "PARES", "JUEGO", "RECUENTO"]
        self.fase_actual = self.fases[0]

        self.manos = {}
        self.cartas_a_descartar = {}
        self.votos_mus = []
        self.historial_apuestas = []
        
        self.hand_size = 4
        self.deck = [(v, s) for v in range(1, 13) for s in range(4) if v != 8 and v != 9]

        # Acciones: 0=pasar, 1=envido, 2=mus, 3=no mus, 4=confirmar, 10-13=descartar carta 0-3
        self.action_spaces = {agent: spaces.Discrete(14) for agent in self.agents}
        
        self.observation_spaces = {
            agent: spaces.Dict({
                "cartas": spaces.Box(low=1, high=12, shape=(self.hand_size, 2), dtype=np.int8),
                "fase": spaces.Discrete(len(self.fases)),
                "turno": spaces.Discrete(4)
            }) for agent in self.agents
        }

        # Añadimos nuevas propiedades para manejar apuestas
        self.apuestas = {
            "GRANDE": {"valor": 0, "retador": None, "ganador": None},
            "CHICA": {"valor": 0, "retador": None, "ganador": None},
            "PARES": {"valor": 0, "retador": None, "ganador": None},
            "JUEGO": {"valor": 0, "retador": None, "ganador": None}
        }
        self.ultimo_que_hablo = None
        self.acciones_validas = {
            "GRANDE": [0, 1, 5],  # 0=pasar, 1=envite, 5=no quiero
            "CHICA": [0, 1, 5],
            "PARES": [0, 1, 5],
            "JUEGO": [0, 1, 5, 6]  # 6=ordago
        }

    def generar_mazo(self):
        self.mazo = [(v, p) for p in range(4) for v in range(1, 13) if v not in [8, 9]]
        random.shuffle(self.mazo)

    def reset(self, seed=None, options=None):
        self.generar_mazo()
        self.cartas_a_descartar = {agent: [] for agent in self.agents}
        self.votos_mus = []
        self.historial_apuestas = []
        
        self.agents = self.possible_agents[:]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        
        self.repartir_cartas()
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.fase_actual = "MUS"

    def repartir_cartas(self):
        self.manos = {agent: [self.mazo.pop() for _ in range(4)] for agent in self.agents}

    def observe(self, agent):
        return {
            "cartas": np.array(self.manos[agent], dtype=np.int8),
            "fase": self.fases.index(self.fase_actual),
            "turno": self.agent_name_mapping[self.agent_selection]
        }
    
    def _was_done_step(self, action):
        # 1. Find the next agent that's not done
        if self.agents:
            self.agent_selection = self.agent_selector.next()
        
        # 2. Optional: You can add additional checks here
        if action is not None:
            assert self.action_spaces[self.agent_selection].contains(action), \
                f"Action {action} is invalid for agent {self.agent_selection}"
        
        # 3. Handle rewards accumulation for done agents
        if self.agent_selection in self.dones:
            self.rewards[self.agent_selection] = 0

    def calcular_puntos(self, mano, fase):
        """Calcula los puntos para cada fase"""
        valores = sorted([carta[0] for carta in mano], reverse=True)
        
        if fase == "GRANDE":
            # Suma de las 3 cartas más altas (sin contar la más baja)
            return sum(valores[:3])
        elif fase == "CHICA":
            # Suma de las 3 cartas más bajas (sin contar la más alta)
            return sum(valores[1:])
        elif fase == "PARES":
            # Calcula pares, medias y duples
            counts = {}
            for v in valores:
                counts[v] = counts.get(v, 0) + 1
            
            if any(c >= 3 for c in counts.values()):
                return 6  # Duples
            elif list(counts.values()).count(2) >= 2:
                return 5  # Medias
            elif any(c == 2 for c in counts.values()):
                return 2  # Pares
            return 0
        elif fase == "JUEGO":
            total = sum(valores)
            return 31 if total == 31 else (32 if total >= 32 else 0)

    
    def step(self, action):
        agent = self.agent_selection
        
        if self.dones[agent]:
            self._was_done_step(action)
            return

        if self.fase_actual == "MUS":
            if action in [2, 3]:  # Mus (2) o No Mus (3)
                if agent not in [a for a, v in self.votos_mus]:
                    self.votos_mus.append((agent, action))
                    self.agent_selection = self.agent_selector.next()
                    
                    if len(self.votos_mus) == len(self.agents):
                        votos = [v for a, v in self.votos_mus]
                        if 3 in votos:  # Si alguien dijo No Mus
                            self.fase_actual = "GRANDE"
                            # Reiniciar el selector de agentes para la nueva fase
                            self.agent_selector = agent_selector(self.agents)
                            self.agent_selection = self.agent_selector.next()
                        else:  # Todos dijeron Mus
                            self.fase_actual = "DESCARTE"
                            self.cartas_a_descartar = {agent: [] for agent in self.agents}
                        self.votos_mus = []
                return

        elif self.fase_actual == "DESCARTE":
            if 10 <= action <= 13:  # Selección de cartas
                carta_idx = action - 10
                if agent not in self.cartas_a_descartar:
                    self.cartas_a_descartar[agent] = []
                    
                if carta_idx in self.cartas_a_descartar[agent]:
                    self.cartas_a_descartar[agent].remove(carta_idx)
                else:
                    self.cartas_a_descartar[agent].append(carta_idx)
                return

            elif action == 4:  # Confirmar descarte
                nuevas_cartas = []
                for i in range(4):
                    if i in self.cartas_a_descartar.get(agent, []):
                        if self.mazo:
                            nuevas_cartas.append(self.mazo.pop())
                        else:
                            nuevas_cartas.append(self.manos[agent][i])
                    else:
                        nuevas_cartas.append(self.manos[agent][i])
                
                self.manos[agent] = nuevas_cartas
                self.cartas_a_descartar[agent] = []
                
                self.agent_selection = self.agent_selector.next()
                
                if all(len(self.cartas_a_descartar.get(ag, [])) == 0 for ag in self.agents):
                    # Cambiado: Avanzar directamente a GRANDE después del descarte
                    self.fase_actual = "MUS"
                    # Reiniciar el selector de agentes para la nueva fase
                    self.agent_selector = agent_selector(self.agents)
                    self.agent_selection = self.agent_selector.next()
                return
        
            
        elif self.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
            self.procesar_apuesta(self.fase_actual, agent, action)
            
            # Verificación adicional para asegurar el avance
            if self.fase_actual in self.apuestas and self.apuestas[self.fase_actual]["ganador"] is not None:
                if self.agent_selector.is_last():  # Solo avanzar cuando todos hayan tenido oportunidad
                    self.avanzar_fase()

    def procesar_apuesta(self, fase, agent, action):
        """Maneja la lógica de apuestas para todas las fases"""
        if action not in self.acciones_validas[fase]:
            return

        apuesta = self.apuestas[fase]
        
        if action == 0:  # Pasar
            if apuesta["valor"] == 0:  # Nadie ha envitado aún
                # Avanzamos al siguiente jugador
                self.agent_selection = self.agent_selector.next()
                print(f"Pasando turno a {self.agent_selection}")
                # Si hemos completado una ronda completa de pasos
                self.avanzar_fase()
            else:  # Alguien envitó antes
                if agent != apuesta["retador"]:
                    # Avanzamos al siguiente jugador
                    self.agent_selection = self.agent_selector.next()
                    
                    # Si hemos completado una ronda completa de pasos después del envite
                    
                    self.avanzar_fase()

        elif action == 1:  # Envite
            apuesta["valor"] += 1
            apuesta["retador"] = agent
            self.ultimo_que_hablo = agent
            # Al hacer un envite, pasamos al siguiente jugador
            self.agent_selection = self.agent_selector.next()

        elif action == 5:  # No quiero
            if apuesta["valor"] > 0:
                apuesta["ganador"] = self.ultimo_que_hablo
                self.rewards[apuesta["ganador"]] += apuesta["valor"]
                self.avanzar_fase()

        elif action == 6:  # Ordago
            apuesta["valor"] = 40
            apuesta["retador"] = agent
            self.ultimo_que_hablo = agent
            # Al hacer ordago, pasamos al siguiente jugador
            self.agent_selection = self.agent_selector.next()

        # Caso especial: si todos han pasado sin envites
        if (action == 0 and apuesta["valor"] == 0 and 
            self.agent_selector.is_last() and self.agent_selection == self.agents[0]):
            self.avanzar_fase()

    def avanzar_fase(self):
        """Avanza a la siguiente fase del juego"""
        current_idx = self.fases.index(self.fase_actual)
        if current_idx < len(self.fases) - 1:
            self.fase_actual = self.fases[current_idx + 1]
            # Reiniciamos completamente el selector de agentes
            self.agent_selector = agent_selector(self.agents)
            self.agent_selection = self.agent_selector.next()
            self.ultimo_que_hablo = None
            
            # Reiniciamos las apuestas para la nueva fase
            if self.fase_actual in self.apuestas:
                self.apuestas[self.fase_actual] = {"valor": 0, "retador": None, "ganador": None}
        else:
            self.fase_actual = "RECUENTO"

    def determinar_ganador_global(self):
        """Determina el ganador final del juego"""
        # Implementar lógica para determinar el ganador global
        # basado en los puntos de cada fase
        pass

    def render(self):
        print(f"Fase: {self.fase_actual}")
        for ag in self.agents:
            print(f"{ag}: {self.manos[ag]} descarta {self.cartas_a_descartar.get(ag, [])}")
        if self.fase_actual == "MUS":
            print(f"Votos MUS: {self.votos_mus}")

    def close(self):
        pass

def env():
    return MusEnv()