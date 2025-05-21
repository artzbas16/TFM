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
        
        # Definir equipos (0,2) y (1,3)
        self.equipos = {
            "equipo_1": ["jugador_0", "jugador_2"],
            "equipo_2": ["jugador_1", "jugador_3"]
        }
        # Mapeo inverso para saber a qué equipo pertenece cada jugador
        self.equipo_de_jugador = {}
        for equipo, jugadores in self.equipos.items():
            for jugador in jugadores:
                self.equipo_de_jugador[jugador] = equipo

        # Fases del juego
        self.fases = ["MUS", "DESCARTE", "GRANDE", "CHICA", "PARES", "JUEGO", "RECUENTO"]
        self.fase_actual = self.fases[0]

        self.manos = {}
        self.cartas_a_descartar = {}
        self.votos_mus = []
        self.historial_apuestas = []
        
        # Diccionarios para almacenar declaraciones
        self.declaraciones_pares = {}
        self.declaraciones_juego = {}
        self.valores_juego = {}  # Para almacenar el valor numérico de cada mano
        
        # Registro de decisiones de los jugadores
        self.ultima_decision = {agent: "Esperando..." for agent in self.agents}
        
        # Control de apuestas
        self.apuesta_actual = 0
        self.equipo_apostador = None
        self.jugador_apostador = None
        self.ronda_completa = False
        self.jugadores_pasado = set()
        
        self.hand_size = 4
        self.deck = [(v, s) for v in range(1, 13) for s in range(4) if v != 8 and v != 9]
        self.mazo = self.deck.copy()  # Inicializar mazo

        # Acciones: 0=pasar, 1=envido, 2=mus, 3=no mus, 4=confirmar, 5=no quiero, 6=ordago, 
        # 7=quiero (capear), 11-14=descartar carta 0-3
        self.action_spaces = {agent: spaces.Discrete(15) for agent in self.agents}
        
        self.observation_spaces = {
            agent: spaces.Dict({
                "cartas": spaces.Box(low=1, high=12, shape=(self.hand_size, 2), dtype=np.int8),
                "fase": spaces.Discrete(len(self.fases)),
                "turno": spaces.Discrete(4)
            }) for agent in self.agents
        }

        # Propiedades para manejar apuestas
        self.apuestas = {
            "GRANDE": {"valor": 0, "retador": None, "ganador": None},
            "CHICA": {"valor": 0, "retador": None, "ganador": None},
            "PARES": {"valor": 0, "retador": None, "ganador": None},
            "JUEGO": {"valor": 0, "retador": None, "ganador": None}
        }
        self.ultimo_que_hablo = None
        self.acciones_validas = {
            "GRANDE": [0, 1, 5, 6, 7],  # 0=pasar, 1=envite, 5=no quiero, 6=ordago, 7=quiero
            "CHICA": [0, 1, 5, 6, 7],
            "PARES": [0, 1, 5, 6, 7],
            "JUEGO": [0, 1, 5, 6, 7]
        }
        
        # Inicializar dones y rewards
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Puntos totales para los equipos
        self.puntos_equipos = {"equipo_1": 0, "equipo_2": 0}
        
        # Ganadores de cada fase para el recuento final
        self.ganadores_fases = {
            "GRANDE": None,
            "CHICA": None,
            "PARES": None,
            "JUEGO": None
        }

    def generar_mazo(self):
        self.mazo = [(v, p) for p in range(4) for v in range(1, 13) if v not in [8, 9]]
        random.shuffle(self.mazo)

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            
        self.generar_mazo()
        self.cartas_a_descartar = {agent: [] for agent in self.agents}
        self.votos_mus = []
        self.historial_apuestas = []
        self.declaraciones_pares = {}
        self.declaraciones_juego = {}
        self.valores_juego = {}
        self.ultima_decision = {agent: "Esperando..." for agent in self.agents}
        
        # Reiniciar control de apuestas
        self.apuesta_actual = 0
        self.equipo_apostador = None
        self.jugador_apostador = None
        self.ronda_completa = False
        self.jugadores_pasado = set()
        
        self.agents = self.possible_agents[:]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        
        self.repartir_cartas()
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.fase_actual = "MUS"
        
        # Reiniciar apuestas
        for fase in self.apuestas:
            self.apuestas[fase] = {"valor": 0, "retador": None, "ganador": None}
            
        # Reiniciar ganadores de fases
        for fase in self.ganadores_fases:
            self.ganadores_fases[fase] = None
            
        # Devolver observación inicial
        return self.observe(self.agent_selection)

    def repartir_cartas(self):
        self.manos = {agent: [self.mazo.pop() for _ in range(4)] for agent in self.agents}
        # Calcular automáticamente las declaraciones y valores
        self.actualizar_declaraciones()

    def actualizar_declaraciones(self):
        """Actualiza automáticamente las declaraciones de pares y juego para todos los jugadores"""
        self.declaraciones_pares = {}
        self.declaraciones_juego = {}
        self.valores_juego = {}
        
        for agent in self.agents:
            # Calcular si tiene pares
            self.declaraciones_pares[agent] = self.tiene_pares(self.manos[agent])
            
            # Calcular si tiene juego y su valor
            valor_juego = self.calcular_valor_juego(self.manos[agent])
            self.valores_juego[agent] = valor_juego
            self.declaraciones_juego[agent] = valor_juego >= 31

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
            assert self.action_spaces[self.agent_selection].contains(action), ''
            f"Action {action} is invalid for agent {self.agent_selection}"
        
        # 3. Handle rewards accumulation for done agents
        if self.agent_selection in self.dones and self.dones[self.agent_selection]:
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
            # Devolver el valor calculado para juego
            return self.calcular_valor_juego(mano)

    def calcular_valor_juego(self, mano):
        """Calcula el valor de la mano para juego (figuras valen 10, las demás su valor)"""
        total = 0
        for valor, _ in mano:
            # Las figuras (10, 11, 12) valen 10 puntos
            if valor >= 10:
                total += 10
            else:
                total += valor
        return total

    def tiene_pares(self, mano):
        """Determina si un jugador tiene pares o no"""
        valores = [carta[0] for carta in mano]
        counts = {}
        for v in valores:
            counts[v] = counts.get(v, 0) + 1
        return any(c >= 2 for c in counts.values())
    
    def puede_hablar(self, agent):
        """Determina si un jugador puede hablar en la fase actual"""
        if self.fase_actual == "PARES":
            return self.declaraciones_pares.get(agent, False)
        elif self.fase_actual == "JUEGO":
            return self.declaraciones_juego.get(agent, False)
        return True
    
    def siguiente_jugador_que_puede_hablar(self):
        """Encuentra el siguiente jugador que puede hablar en la fase actual"""
        # Guardar el jugador actual
        jugador_actual = self.agent_selection
        
        # Buscar el siguiente jugador que puede hablar
        for _ in range(len(self.agents)):
            self.agent_selection = self.agent_selector.next()
            if self.puede_hablar(self.agent_selection):
                return
        
        # Si nadie puede hablar, avanzar a la siguiente fase
        if self.fase_actual in ["PARES", "JUEGO"]:
            self.avanzar_fase()
        else:
            # Si no estamos en PARES o JUEGO, restaurar el jugador actual
            self.agent_selection = jugador_actual
    
    def es_del_mismo_equipo(self, jugador1, jugador2):
        """Determina si dos jugadores son del mismo equipo"""
        return self.equipo_de_jugador[jugador1] == self.equipo_de_jugador[jugador2]
    
    def step(self, action):
        agent = self.agent_selection
        
        if self.dones[agent]:
            self._was_done_step(action)
            return
            
        # Verificar que la acción es válida
        if not self.action_spaces[agent].contains(action):
            print(f"Acción inválida {action} para el agente {agent}")
            return
            
        # Registrar la decisión del jugador
        self.registrar_decision(agent, action)

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
                            # Reiniciar control de apuestas
                            self.apuesta_actual = 0
                            self.equipo_apostador = None
                            self.jugador_apostador = None
                            self.ronda_completa = False
                            self.jugadores_pasado = set()
                        else:  # Todos dijeron Mus
                            self.fase_actual = "DESCARTE"
                            self.cartas_a_descartar = {agent: [] for agent in self.agents}
                        self.votos_mus = []
                return

        elif self.fase_actual == "DESCARTE":
            if 11 <= action <= 14:  # Selección de cartas (ajustado por las nuevas acciones)
                carta_idx = action - 11
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
                
                # Actualizar declaraciones después del descarte
                self.actualizar_declaraciones()
                
                self.agent_selection = self.agent_selector.next()
                
                # Verificar si todos han descartado
                if all(len(self.cartas_a_descartar.get(ag, [])) == 0 for ag in self.agents):
                    # Después del descarte, volver a la fase MUS
                    self.fase_actual = "MUS"
                    # Reiniciar el selector de agentes para la nueva fase
                    self.agent_selector = agent_selector(self.agents)
                    self.agent_selection = self.agent_selector.next()
                return
            
        elif self.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
            # Verificar si el jugador puede hablar en esta fase
            if not self.puede_hablar(agent) and self.fase_actual in ["PARES", "JUEGO"]:
                # Si no puede hablar, pasar al siguiente jugador que pueda
                self.siguiente_jugador_que_puede_hablar()
                return
                
            # Procesar la acción según la lógica correcta de apuestas
            self.procesar_apuesta_corregida(self.fase_actual, agent, action)
                    
        # Devolver observación actualizada
        return self.observe(self.agent_selection)
        
    def registrar_decision(self, agent, action):
        """Registra la decisión tomada por un jugador"""
        if action == 0:
            self.ultima_decision[agent] = "Paso"
        elif action == 1:
            self.ultima_decision[agent] = "Envido"
        elif action == 2:
            self.ultima_decision[agent] = "Mus"
        elif action == 3:
            self.ultima_decision[agent] = "No Mus"
        elif action == 4:
            self.ultima_decision[agent] = "Confirmar"
        elif action == 5:
            self.ultima_decision[agent] = "No quiero"
        elif action == 6:
            self.ultima_decision[agent] = "Órdago"
        elif action == 7:
            self.ultima_decision[agent] = "Quiero"
        elif 11 <= action <= 14:
            carta_idx = action - 11
            if carta_idx in self.cartas_a_descartar.get(agent, []):
                self.ultima_decision[agent] = f"Deseleccionar carta {carta_idx+1}"
            else:
                self.ultima_decision[agent] = f"Seleccionar carta {carta_idx+1}"

    def procesar_apuesta_corregida(self, fase, agent, action):
        """Implementa la lógica correcta de apuestas para el juego de Mus"""
        if action not in self.acciones_validas[fase]:
            print(f"Acción {action} no válida para la fase {fase}")
            return
            
        # Si es la primera acción de la fase, reiniciar el control de apuestas
        if self.apuesta_actual == 0 and self.equipo_apostador is None:
            self.jugadores_pasado = set()
            
        # Procesar según la acción
        if action == 0:  # Pasar
            # Registrar que este jugador ha pasado
            self.jugadores_pasado.add(agent)
            
            # Si hay una apuesta activa y todos los jugadores del equipo contrario han pasado
            if self.apuesta_actual > 0 and self.equipo_apostador is not None:
                equipo_contrario = "equipo_2" if self.equipo_apostador == "equipo_1" else "equipo_1"
                jugadores_equipo_contrario = self.equipos[equipo_contrario]
                
                if all(jugador in self.jugadores_pasado for jugador in jugadores_equipo_contrario):
                    # El equipo apostador gana la apuesta
                    self.puntos_equipos[self.equipo_apostador] += self.apuesta_actual
                    self.ganadores_fases[fase] = self.equipo_apostador
                    
                    # Avanzar a la siguiente fase
                    self.avanzar_fase()
                    return
            
            # Si todos los jugadores han pasado sin que haya apuesta
            if len(self.jugadores_pasado) == len(self.agents) and self.apuesta_actual == 0:
                # Determinar el ganador de la fase por puntos
                self.determinar_ganador_fase(fase)
                
                # Avanzar a la siguiente fase
                self.avanzar_fase()
                return
                
            # Pasar al siguiente jugador
            self.agent_selection = self.agent_selector.next()
            
        elif action == 1:  # Envido
            # Aumentar la apuesta
            self.apuesta_actual += 1
            
            # Registrar quién ha apostado
            self.jugador_apostador = agent
            self.equipo_apostador = self.equipo_de_jugador[agent]
            
            # Reiniciar los jugadores que han pasado
            self.jugadores_pasado = set()
            
            # Pasar al siguiente jugador
            self.agent_selection = self.agent_selector.next()
            
        elif action == 5:  # No quiero
            if self.apuesta_actual > 0 and self.equipo_apostador is not None:
                # Verificar que el jugador es del equipo contrario al apostador
                if not self.es_del_mismo_equipo(agent, self.jugador_apostador):
                    # El equipo apostador gana la apuesta
                    self.puntos_equipos[self.equipo_apostador] += self.apuesta_actual
                    self.ganadores_fases[fase] = self.equipo_apostador
                    
                    # Avanzar a la siguiente fase
                    self.avanzar_fase()
                    return
            
            # Si no hay apuesta o el jugador es del mismo equipo, simplemente pasar turno
            self.agent_selection = self.agent_selector.next()
            
        elif action == 6:  # Ordago
            # Establecer una apuesta muy alta
            self.apuesta_actual = 40
            
            # Registrar quién ha apostado
            self.jugador_apostador = agent
            self.equipo_apostador = self.equipo_de_jugador[agent]
            
            # Reiniciar los jugadores que han pasado
            self.jugadores_pasado = set()
            
            # Pasar al siguiente jugador
            self.agent_selection = self.agent_selector.next()
            
        elif action == 7:  # Quiero (capear)
            if self.apuesta_actual > 0 and self.equipo_apostador is not None:
                # Verificar que el jugador es del equipo contrario al apostador
                if not self.es_del_mismo_equipo(agent, self.jugador_apostador):
                    # Registrar que este jugador ha aceptado la apuesta
                    self.jugadores_pasado.add(agent)
                    
                    # Verificar si todos los jugadores del equipo contrario han aceptado o pasado
                    equipo_contrario = "equipo_2" if self.equipo_apostador == "equipo_1" else "equipo_1"
                    jugadores_equipo_contrario = self.equipos[equipo_contrario]
                    
                    if all(jugador in self.jugadores_pasado for jugador in jugadores_equipo_contrario):
                        # Determinar el ganador por puntos
                        self.determinar_ganador_fase(fase)
                        
                        # Avanzar a la siguiente fase
                        self.avanzar_fase()
                        return
            
            # Pasar al siguiente jugador
            self.agent_selection = self.agent_selector.next()
    
    def determinar_ganador_fase(self, fase):
        """Determina el ganador de una fase basado en los puntos"""
        # Calcular los puntos de cada equipo en esta fase
        puntos_equipos = {"equipo_1": 0, "equipo_2": 0}
        
        for agent in self.agents:
            equipo = self.equipo_de_jugador[agent]
            puntos = self.calcular_puntos(self.manos[agent], fase)
            
            # En GRANDE y JUEGO, gana el que tiene más puntos
            if fase in ["GRANDE", "JUEGO"]:
                if puntos > puntos_equipos[equipo]:
                    puntos_equipos[equipo] = puntos
            # En CHICA, gana el que tiene menos puntos (pero no cero)
            elif fase == "CHICA":
                if puntos_equipos[equipo] == 0 or (puntos > 0 and puntos < puntos_equipos[equipo]):
                    puntos_equipos[equipo] = puntos if puntos > 0 else 999
            # En PARES, gana el que tiene mejor combinación
            elif fase == "PARES":
                if puntos > puntos_equipos[equipo]:
                    puntos_equipos[equipo] = puntos
        
        # Determinar el equipo ganador
        if fase == "CHICA":
            # En CHICA, gana el que tiene menos puntos
            if puntos_equipos["equipo_1"] == 999:
                puntos_equipos["equipo_1"] = 0
            if puntos_equipos["equipo_2"] == 999:
                puntos_equipos["equipo_2"] = 0
                
            if puntos_equipos["equipo_1"] == 0 and puntos_equipos["equipo_2"] == 0:
                # Si nadie tiene puntos, no hay ganador
                self.ganadores_fases[fase] = None
            elif puntos_equipos["equipo_1"] == 0:
                # Si equipo_1 no tiene puntos, gana equipo_2
                self.ganadores_fases[fase] = "equipo_2"
                self.puntos_equipos["equipo_2"] += 1
            elif puntos_equipos["equipo_2"] == 0:
                # Si equipo_2 no tiene puntos, gana equipo_1
                self.ganadores_fases[fase] = "equipo_1"
                self.puntos_equipos["equipo_1"] += 1
            elif puntos_equipos["equipo_1"] < puntos_equipos["equipo_2"]:
                self.ganadores_fases[fase] = "equipo_1"
                self.puntos_equipos["equipo_1"] += 1
            elif puntos_equipos["equipo_2"] < puntos_equipos["equipo_1"]:
                self.ganadores_fases[fase] = "equipo_2"
                self.puntos_equipos["equipo_2"] += 1
            else:
                # Empate, no hay ganador
                self.ganadores_fases[fase] = None
        else:
            # En las demás fases, gana el que tiene más puntos
            if puntos_equipos["equipo_1"] > puntos_equipos["equipo_2"]:
                self.ganadores_fases[fase] = "equipo_1"
                self.puntos_equipos["equipo_1"] += 1
            elif puntos_equipos["equipo_2"] > puntos_equipos["equipo_1"]:
                self.ganadores_fases[fase] = "equipo_2"
                self.puntos_equipos["equipo_2"] += 1
            else:
                # Empate, no hay ganador
                self.ganadores_fases[fase] = None

    def avanzar_fase(self):
        """Avanza a la siguiente fase del juego"""
        current_idx = self.fases.index(self.fase_actual)
        
        # Determinar la siguiente fase de manera más segura
        if current_idx < len(self.fases) - 1:
            # Por defecto, avanzamos a la siguiente fase
            next_idx = current_idx + 1
            next_fase = self.fases[next_idx]
            
            # Lógica para saltar fases
            if next_fase == "PARES" and not any(self.declaraciones_pares.values()):
                # Si nadie tiene pares, saltamos a JUEGO
                if "JUEGO" in self.fases:
                    next_fase = "JUEGO"
                else:
                    # Si no hay fase JUEGO, saltamos a RECUENTO
                    next_fase = "RECUENTO"
            
            if next_fase == "JUEGO" and not any(self.declaraciones_juego.values()):
                # Si nadie tiene juego, saltamos a RECUENTO
                next_fase = "RECUENTO"
                
            self.fase_actual = next_fase
            
            # Reiniciamos completamente el selector de agentes para la nueva fase
            self.agent_selector = agent_selector(self.agents)
            self.agent_selection = self.agent_selector.next()
            
            # Reiniciar control de apuestas para la nueva fase
            self.apuesta_actual = 0
            self.equipo_apostador = None
            self.jugador_apostador = None
            self.ronda_completa = False
            self.jugadores_pasado = set()
                
            # Si llegamos a la fase de RECUENTO, determinar el ganador
            if self.fase_actual == "RECUENTO":
                self.determinar_ganador_global()
        else:
            self.fase_actual = "RECUENTO"
            # Marcar el juego como terminado
            for agent in self.agents:
                self.dones[agent] = True
            # Determinar ganador final
            self.determinar_ganador_global()

    def determinar_ganador_global(self):
        """Determina el ganador final del juego"""
        # El ganador es el equipo con más puntos
        equipo_ganador = max(self.puntos_equipos.items(), key=lambda x: x[1])[0]
        
        # Asignar recompensa final a los jugadores del equipo ganador
        for agent in self.agents:
            if self.equipo_de_jugador[agent] == equipo_ganador:
                self.rewards[agent] = sum(self.puntos_equipos.values())
            else:
                self.rewards[agent] = -self.puntos_equipos[self.equipo_de_jugador[agent]]
        
        return equipo_ganador

    def render(self):
        print(f"Fase: {self.fase_actual}")
        for ag in self.agents:
            print(f"{ag}: {self.manos[ag]} descarta {self.cartas_a_descartar.get(ag, [])}")
        if self.fase_actual == "MUS":
            print(f"Votos MUS: {self.votos_mus}")
        elif self.fase_actual == "PARES":
            print(f"Declaraciones PARES: {self.declaraciones_pares}")
        elif self.fase_actual == "JUEGO":
            print(f"Declaraciones JUEGO: {self.declaraciones_juego}")
            print(f"Valores JUEGO: {self.valores_juego}")
        elif self.fase_actual == "RECUENTO":
            print(f"Puntos equipos: {self.puntos_equipos}")
            print(f"Ganadores fases: {self.ganadores_fases}")

    def close(self):
        pass

def env():
    return MusEnv()