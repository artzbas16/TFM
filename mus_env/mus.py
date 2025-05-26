from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces
import numpy as np
import random
import time

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
        self.valores_juego = {}
        
        # Registro de decisiones de los jugadores
        self.ultima_decision = {agent: "Esperando..." for agent in self.agents}
        
        # Control de apuestas
        self.apuesta_actual = 0
        self.equipo_apostador = None
        self.jugador_apostador = None
        self.ronda_completa = False
        self.jugadores_pasado = set()
        self.jugadores_hablaron = set()
        self.jugadores_que_pueden_hablar = set()
        self.hay_ordago = False
        self.jugadores_confirmaron_descarte = set()
        
        self.hand_size = 4
        # Crear mazo sin 8s y 9s correctamente
        self.deck = [(v, s) for v in range(1, 13) for s in range(4) if v not in [8, 9]]
        self.mazo = self.deck.copy()

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

        # CORRECCIÓN: Estructura de apuestas que coincida con la GUI
        self.apuestas = {
            "equipo_1": {
                "GRANDE": 0,
                "CHICA": 0,
                "PARES": 0,
                "JUEGO": 0
            },
            "equipo_2": {
                "GRANDE": 0,
                "CHICA": 0,
                "PARES": 0,
                "JUEGO": 0
            }
        }
        
        self.acciones_validas = {
            "GRANDE": [0, 1, 5, 6, 7],
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

        self.partidas_ganadas = {"equipo_1": 0, "equipo_2": 0}  # Nuevo: registro de partidas ganadas
        self.partida_terminada = False  # Nuevo: indica si la partida actual terminó
        
        # Ganadores de cada fase para el recuento final
        self.ganadores_fases = {
            "GRANDE": None,
            "CHICA": None,
            "PARES": None,
            "JUEGO": None
        }

        # Control de tiempo para las acciones
        self.action_delay = 1.0  # 1 segundo de delay
        self.last_action_time = 0

    def generar_mazo(self):
        """Generar mazo correctamente sin 8s y 9s"""
        self.mazo = [(v, p) for p in range(4) for v in range(1, 13) if v not in [8, 9]]
        random.shuffle(self.mazo)

    def reset(self, seed=75, options=None):
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

        self.puntos_equipos = {"equipo_1": 0, "equipo_2": 0}
        self.partida_terminada = False
        
        # Reiniciar control de apuestas
        self.apuesta_actual = 0
        self.equipo_apostador = None
        self.jugador_apostador = None
        self.ronda_completa = False
        self.jugadores_pasado = set()
        self.jugadores_hablaron = set()
        self.jugadores_que_pueden_hablar = set()
        self.hay_ordago = False
        
        self.agents = self.possible_agents[:]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        
        self.repartir_cartas()
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.fase_actual = "MUS"
        
        # Reiniciar apuestas
        for equipo in self.apuestas:
            for fase in self.apuestas[equipo]:
                self.apuestas[equipo][fase] = 0
            
        # Reiniciar ganadores de fases
        for fase in self.ganadores_fases:
            self.ganadores_fases[fase] = None
            
        # Reiniciar tiempo
        self.last_action_time = time.time()
            
        return self.observe(self.agent_selection)

    def repartir_cartas(self):
        """Verificar que hay suficientes cartas antes de repartir"""
        if len(self.mazo) < len(self.agents) * self.hand_size:
            self.generar_mazo()
            
        self.manos = {}
        for agent in self.agents:
            self.manos[agent] = []
            for _ in range(self.hand_size):
                if self.mazo:
                    self.manos[agent].append(self.mazo.pop())
                else:
                    self.generar_mazo()
                    self.manos[agent].append(self.mazo.pop())
        
        self.actualizar_declaraciones()

    def actualizar_declaraciones(self):
        """Actualiza automáticamente las declaraciones de pares y juego para todos los jugadores"""
        self.declaraciones_pares = {}
        self.declaraciones_juego = {}
        self.valores_juego = {}
        
        for agent in self.agents:
            if agent in self.manos:
                # Calcular si tiene pares
                self.declaraciones_pares[agent] = self.tiene_pares(self.manos[agent])
                
                # Calcular si tiene juego y su valor
                valor_juego = self.calcular_valor_juego(self.manos[agent])
                self.valores_juego[agent] = valor_juego
                self.declaraciones_juego[agent] = valor_juego >= 31

    def actualizar_jugadores_que_pueden_hablar(self):
        """Mejorar la lógica de quién puede hablar"""
        self.jugadores_que_pueden_hablar = set()
        self.equipos_que_pueden_hablar = set()
        
        if self.fase_actual == "PARES":
            for agent in self.agents:
                if self.declaraciones_pares.get(agent, False):
                    self.jugadores_que_pueden_hablar.add(agent)
                    self.equipos_que_pueden_hablar.add(self.equipo_de_jugador[agent])
        elif self.fase_actual == "JUEGO":
            for agent in self.agents:
                valor_juego = self.calcular_valor_juego(self.manos[agent])
                # Permitir hablar si tiene juego (>=31) o si nadie tiene juego (todos pueden hablar)
                if valor_juego >= 31 or not any(self.declaraciones_juego.values()):
                    self.jugadores_que_pueden_hablar.add(agent)
                    self.equipos_que_pueden_hablar.add(self.equipo_de_jugador[agent])
        else:
            # En otras fases, todos pueden hablar
            self.jugadores_que_pueden_hablar = set(self.agents)
            self.equipos_que_pueden_hablar = {"equipo_1", "equipo_2"}
            
        # Si solo un equipo puede hablar, ese equipo gana automáticamente
        if self.fase_actual in ["PARES", "JUEGO"] and len(self.equipos_que_pueden_hablar) == 1:
            equipo_ganador = list(self.equipos_que_pueden_hablar)[0]
            self.ganadores_fases[self.fase_actual] = equipo_ganador
            
            # Asignar puntos según la fase
            if self.fase_actual == "PARES":
                puntos = self.calcular_puntos_pares(equipo_ganador)
            elif self.fase_actual == "JUEGO":
                puntos = self.calcular_puntos_juego(equipo_ganador)
            else:
                puntos = 1
                
            self.puntos_equipos[equipo_ganador] += puntos
            self.apuestas[equipo_ganador][self.fase_actual] = puntos
            print(f"Equipo {equipo_ganador} gana {puntos} puntos en {self.fase_actual} automáticamente")
            
            self.avanzar_fase()
        elif self.fase_actual in ["PARES", "JUEGO"] and len(self.equipos_que_pueden_hablar) == 0:
            print(f"Nadie puede hablar en {self.fase_actual}, avanzando...")
            self.avanzar_fase()

    def observe(self, agent):
        """Verificar que el agente existe en las manos"""
        if agent not in self.manos:
            return {
                "cartas": np.zeros((self.hand_size, 2), dtype=np.int8),
                "fase": self.fases.index(self.fase_actual),
                "turno": self.agent_name_mapping.get(self.agent_selection, 0)
            }
        
        return {
            "cartas": np.array(self.manos[agent], dtype=np.int8),
            "fase": self.fases.index(self.fase_actual),
            "turno": self.agent_name_mapping.get(self.agent_selection, 0)
        }
    
    def _was_done_step(self, action):
        """Mejorar manejo de agentes terminados"""
        if self.agents:
            attempts = 0
            while attempts < len(self.agents):
                self.agent_selection = self.agent_selector.next()
                if not self.dones.get(self.agent_selection, False):
                    break
                attempts += 1
        
        if action is not None and self.agent_selection in self.action_spaces:
            assert self.action_spaces[self.agent_selection].contains(action), \
                f"Action {action} is invalid for agent {self.agent_selection}"
        
        if self.agent_selection in self.dones and self.dones[self.agent_selection]:
            self.rewards[self.agent_selection] = 0

    def wait_for_action_delay(self):
        """Esperar el tiempo necesario entre acciones"""
        current_time = time.time()
        time_since_last_action = current_time - self.last_action_time
        
        if time_since_last_action < self.action_delay:
            time.sleep(self.action_delay - time_since_last_action)
        
        self.last_action_time = time.time()

    def calcular_puntos(self, mano, fase):
        """Mejorar cálculo de puntos"""
        if not mano:
            return 0
            
        valores = sorted([carta[0] for carta in mano], reverse=True)
        
        if fase == "GRANDE":
            return sum(valores[:3])
        elif fase == "CHICA":
            return sum(sorted(valores)[:3])
        elif fase == "PARES":
            counts = {}
            for v in valores:
                counts[v] = counts.get(v, 0) + 1
            
            if any(c >= 4 for c in counts.values()):
                return 6
            elif any(c >= 3 for c in counts.values()):
                return 3
            elif list(counts.values()).count(2) >= 2:
                return 2
            elif any(c == 2 for c in counts.values()):
                return 1
            return 0
        elif fase == "JUEGO":
            valor_juego = self.calcular_valor_juego(mano)
            if valor_juego >= 31:
                return valor_juego
            else:
                return -(30 - valor_juego)

    def calcular_valor_juego(self, mano):
        """Calcula el valor de la mano para juego"""
        if not mano:
            return 0
            
        total = 0
        for valor, _ in mano:
            if valor >= 10:
                total += 10
            else:
                total += valor
        return total

    def tiene_pares(self, mano):
        """Determina si un jugador tiene pares"""
        if not mano:
            return False
            
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
        """Mejorar búsqueda del siguiente jugador"""
        self.actualizar_jugadores_que_pueden_hablar()
        
        if not self.jugadores_que_pueden_hablar:
            print(f"Nadie puede hablar en la fase {self.fase_actual}, avanzando...")
            self.avanzar_fase()
            return
        
        intentos = 0
        while intentos < len(self.agents) * 2:
            self.agent_selection = self.agent_selector.next()
            if (self.agent_selection in self.jugadores_que_pueden_hablar and 
                not self.dones.get(self.agent_selection, False)):
                return
            intentos += 1
        
        print(f"No se encontró jugador válido en fase {self.fase_actual}, avanzando...")
        self.avanzar_fase()
    
    def es_del_mismo_equipo(self, jugador1, jugador2):
        """Determina si dos jugadores son del mismo equipo"""
        return self.equipo_de_jugador.get(jugador1) == self.equipo_de_jugador.get(jugador2)
    
    def step(self, action):
        """Mejorar manejo de pasos y validaciones con delay"""
        # Aplicar delay antes de procesar la acción
        self.wait_for_action_delay()
        
        agent = self.agent_selection
        
        if self.dones.get(agent, False):
            self._was_done_step(action)
            return
            
        if not self.action_spaces[agent].contains(action):
            print(f"Acción inválida {action} para el agente {agent}")
            return
            
        self.registrar_decision(agent, action)

        if self.fase_actual == "MUS":
            if action in [2, 3]:  # Mus (2) o No Mus (3)
                if action == 3:  # No Mus
                    self.fase_actual = "GRANDE"
                    self.reiniciar_para_nueva_fase()
                    return
                else:  # Mus (2)
                    # Verificar si ya votó
                    if agent not in [a for a, v in self.votos_mus]:
                        self.votos_mus.append((agent, action))
                        
                        if len(self.votos_mus) == len(self.agents):
                            # Todos dijeron Mus
                            self.fase_actual = "DESCARTE"
                            self.cartas_a_descartar = {agent: [] for agent in self.agents}
                            self.votos_mus = []
                            # CORRECCIÓN: Reiniciar selector de agentes para la fase de descarte
                            self.agent_selector = agent_selector(self.agents)
                            self.agent_selection = self.agent_selector.next()
                            # Agregar control para saber quién ha confirmado su descarte
                            self.jugadores_confirmaron_descarte = set()
                            return
                        else:
                            self.agent_selection = self.agent_selector.next()
                    return

        elif self.fase_actual == "DESCARTE":
            if 11 <= action <= 14:  # Selección de cartas
                carta_idx = action - 11
                if agent not in self.cartas_a_descartar:
                    self.cartas_a_descartar[agent] = []
                    
                if carta_idx in self.cartas_a_descartar[agent]:
                    self.cartas_a_descartar[agent].remove(carta_idx)
                else:
                    self.cartas_a_descartar[agent].append(carta_idx)
                return

            elif action == 4:  # Confirmar descarte
                self.realizar_descarte(agent)
                self.jugadores_confirmaron_descarte.add(agent)
                
                # Verificar si todos han confirmado su descarte
                if len(self.jugadores_confirmaron_descarte) == len(self.agents):
                    self.fase_actual = "MUS"
                    self.reiniciar_para_nueva_fase()
                    # Limpiar el conjunto de confirmaciones
                    self.jugadores_confirmaron_descarte = set()
                else:
                    self.agent_selection = self.agent_selector.next()
                return
            
        elif self.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
            self.actualizar_jugadores_que_pueden_hablar()
            
            if agent not in self.jugadores_que_pueden_hablar:
                self.siguiente_jugador_que_puede_hablar()
                return
                
            self.procesar_apuesta_corregida(self.fase_actual, agent, action)
                    
        return self.observe(self.agent_selection)

    def reiniciar_para_nueva_fase(self):
        """Función auxiliar para reiniciar estado entre fases"""
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        self.apuesta_actual = 0
        self.equipo_apostador = None
        self.jugador_apostador = None
        self.ronda_completa = False
        self.jugadores_pasado = set()
        self.jugadores_hablaron = set()
        self.hay_ordago = False
        self.actualizar_jugadores_que_pueden_hablar()
        # Limpiar confirmaciones de descarte
        if hasattr(self, 'jugadores_confirmaron_descarte'):
            self.jugadores_confirmaron_descarte = set()

    def realizar_descarte(self, agent):
        """Mejorar lógica de descarte"""
        if agent not in self.manos:
            return
            
        nuevas_cartas = []
        for i in range(4):
            if i in self.cartas_a_descartar.get(agent, []):
                if self.mazo:
                    nuevas_cartas.append(self.mazo.pop())
                else:
                    self.generar_mazo()
                    if self.mazo:
                        nuevas_cartas.append(self.mazo.pop())
                    else:
                        nuevas_cartas.append(self.manos[agent][i])
            else:
                nuevas_cartas.append(self.manos[agent][i])
        
        self.manos[agent] = nuevas_cartas
        self.cartas_a_descartar[agent] = []
        self.actualizar_declaraciones()
        
    def registrar_decision(self, agent, action):
        """Registra la decisión tomada por un jugador"""
        decisiones = {
            0: "Paso", 1: "Envido", 2: "Mus", 3: "No Mus", 4: "Confirmar",
            5: "No quiero", 6: "Órdago", 7: "Quiero"
        }
        
        if action in decisiones:
            self.ultima_decision[agent] = decisiones[action]
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
        
        self.jugadores_hablaron.add(agent)
        equipo_actual = self.equipo_de_jugador[agent]
        
        if action == 0:  # Pasar
            self.jugadores_pasado.add(agent)
            
            if self.jugadores_hablaron >= self.jugadores_que_pueden_hablar:
                if self.apuesta_actual == 0:
                    self.determinar_ganador_fase(fase)
                    self.avanzar_fase()
                    return
                elif self.equipo_apostador:
                    equipo_contrario = "equipo_2" if self.equipo_apostador == "equipo_1" else "equipo_1"
                    jugadores_equipo_contrario = set(self.equipos[equipo_contrario]) & self.jugadores_que_pueden_hablar
                    
                    if jugadores_equipo_contrario.issubset(self.jugadores_pasado):
                        self.puntos_equipos[self.equipo_apostador] += self.apuesta_actual
                        self.apuestas[self.equipo_apostador][fase] = self.apuesta_actual
                        self.ganadores_fases[fase] = self.equipo_apostador
                        self.avanzar_fase()
                        return
            
            self.siguiente_jugador_que_puede_hablar()
        
        elif action == 1:  # Envido
            self.apuesta_actual += 2
            self.jugador_apostador = agent
            self.equipo_apostador = self.equipo_de_jugador[agent]
            self.jugadores_pasado = set()
            self.jugadores_hablaron = set()
            self.jugadores_hablaron.add(agent)
            self.siguiente_jugador_que_puede_hablar()
        
        elif action == 5:  # No quiero
            if self.apuesta_actual > 0 and self.equipo_apostador is not None:
                if equipo_actual != self.equipo_apostador:
                    puntos_ganados = max(1, self.apuesta_actual - 1)
                    self.puntos_equipos[self.equipo_apostador] += puntos_ganados
                    self.apuestas[self.equipo_apostador][fase] = puntos_ganados
                    self.ganadores_fases[fase] = self.equipo_apostador
                    self.avanzar_fase()
                    return
            
            self.siguiente_jugador_que_puede_hablar()
        
        elif action == 6:  # Ordago
            self.apuesta_actual = 40
            self.hay_ordago = True
            self.jugador_apostador = agent
            self.equipo_apostador = self.equipo_de_jugador[agent]
            self.jugadores_pasado = set()
            self.jugadores_hablaron = set()
            self.jugadores_hablaron.add(agent)
            self.siguiente_jugador_que_puede_hablar()
        
        elif action == 7:  # Quiero
            if self.hay_ordago:
                self.determinar_ganador_fase(fase)
                equipo_ganador = self.ganadores_fases[fase]
                if equipo_ganador:
                    self.puntos_equipos[equipo_ganador] = 30  # Ganar la partida (30 puntos)
                    self.partidas_ganadas[equipo_ganador] += 1  # Registrar partida ganada
                    self.partida_terminada = True
                    self.fase_actual = "RECUENTO"
                    for agent in self.agents:
                        self.dones[agent] = True
                return
            elif self.apuesta_actual > 0 and self.equipo_apostador is not None:
                if equipo_actual != self.equipo_apostador:
                    self.determinar_ganador_fase(fase)
                    self.avanzar_fase()
                    return
            
            self.siguiente_jugador_que_puede_hablar()
    
    def determinar_ganador_fase(self, fase):
        """Determina el ganador de una fase basado en los puntos"""
        jugadores_participantes = self.jugadores_que_pueden_hablar if self.jugadores_que_pueden_hablar else set(self.agents)
        
        puntos_equipos = {"equipo_1": 0, "equipo_2": 0}
        
        for agent in jugadores_participantes:
            equipo = self.equipo_de_jugador[agent]
            puntos = self.calcular_puntos(self.manos[agent], fase)
            
            if fase in ["GRANDE", "JUEGO"]:
                if puntos > puntos_equipos[equipo]:
                    puntos_equipos[equipo] = puntos
            elif fase == "CHICA":
                if puntos_equipos[equipo] == 0 or (puntos > 0 and puntos < puntos_equipos[equipo]):
                    puntos_equipos[equipo] = puntos if puntos > 0 else 999
            elif fase == "PARES":
                if puntos > puntos_equipos[equipo]:
                    puntos_equipos[equipo] = puntos
        
        # Determinar el equipo ganador
        if fase == "CHICA":
            if puntos_equipos["equipo_1"] == 999:
                puntos_equipos["equipo_1"] = 0
            if puntos_equipos["equipo_2"] == 999:
                puntos_equipos["equipo_2"] = 0
                
            if puntos_equipos["equipo_1"] == 0 and puntos_equipos["equipo_2"] == 0:
                self.ganadores_fases[fase] = None
            elif puntos_equipos["equipo_1"] == 0:
                self.ganadores_fases[fase] = "equipo_2"
                puntos_ganados = self.apuesta_actual if self.apuesta_actual > 0 else 1
                self.puntos_equipos["equipo_2"] += puntos_ganados
                self.apuestas["equipo_2"][fase] = puntos_ganados
            elif puntos_equipos["equipo_2"] == 0:
                self.ganadores_fases[fase] = "equipo_1"
                puntos_ganados = self.apuesta_actual if self.apuesta_actual > 0 else 1
                self.puntos_equipos["equipo_1"] += puntos_ganados
                self.apuestas["equipo_1"][fase] = puntos_ganados
            elif puntos_equipos["equipo_1"] < puntos_equipos["equipo_2"]:
                self.ganadores_fases[fase] = "equipo_1"
                puntos_ganados = self.apuesta_actual if self.apuesta_actual > 0 else 1
                self.puntos_equipos["equipo_1"] += puntos_ganados
                self.apuestas["equipo_1"][fase] = puntos_ganados
            elif puntos_equipos["equipo_2"] < puntos_equipos["equipo_1"]:
                self.ganadores_fases[fase] = "equipo_2"
                puntos_ganados = self.apuesta_actual if self.apuesta_actual > 0 else 1
                self.puntos_equipos["equipo_2"] += puntos_ganados
                self.apuestas["equipo_2"][fase] = puntos_ganados
            else:
                self.ganadores_fases[fase] = None
        elif fase == "JUEGO":
            # Si nadie tiene juego (>=31), jugar al punto (quien se acerca más a 30)
            if not any(self.declaraciones_juego.values()):
                # Calcular diferencia respecto a 30 para cada jugador
                diferencias = {}
                for agent in jugadores_participantes:
                    valor = self.calcular_valor_juego(self.manos[agent])
                    diferencias[agent] = abs(30 - valor)
                
                # Encontrar la menor diferencia por equipo
                min_diff_equipo1 = min(diferencias.get(a, 999) for a in self.equipos["equipo_1"] if a in jugadores_participantes)
                min_diff_equipo2 = min(diferencias.get(a, 999) for a in self.equipos["equipo_2"] if a in jugadores_participantes)
                
                if min_diff_equipo1 < min_diff_equipo2:
                    self.ganadores_fases[fase] = "equipo_1"
                    puntos_ganados = self.apuesta_actual if self.apuesta_actual > 0 else 1
                    self.puntos_equipos["equipo_1"] += puntos_ganados
                    self.apuestas["equipo_1"][fase] = puntos_ganados
                elif min_diff_equipo2 < min_diff_equipo1:
                    self.ganadores_fases[fase] = "equipo_2"
                    puntos_ganados = self.apuesta_actual if self.apuesta_actual > 0 else 1
                    self.puntos_equipos["equipo_2"] += puntos_ganados
                    self.apuestas["equipo_2"][fase] = puntos_ganados
                else:
                    self.ganadores_fases[fase] = None
            else:
                # Lógica original para cuando alguien tiene juego
                if puntos_equipos["equipo_1"] > puntos_equipos["equipo_2"]:
                    self.ganadores_fases[fase] = "equipo_1"
                    puntos_adicionales = self.calcular_puntos_juego("equipo_1")
                    puntos_ganados = (self.apuesta_actual if self.apuesta_actual > 0 else 1) + puntos_adicionales
                    self.puntos_equipos["equipo_1"] += puntos_ganados
                    self.apuestas["equipo_1"][fase] = puntos_ganados
                elif puntos_equipos["equipo_2"] > puntos_equipos["equipo_1"]:
                    self.ganadores_fases[fase] = "equipo_2"
                    puntos_adicionales = self.calcular_puntos_juego("equipo_2")
                    puntos_ganados = (self.apuesta_actual if self.apuesta_actual > 0 else 1) + puntos_adicionales
                    self.puntos_equipos["equipo_2"] += puntos_ganados
                    self.apuestas["equipo_2"][fase] = puntos_ganados
                else:
                    self.ganadores_fases[fase] = None
        else:
            if puntos_equipos["equipo_1"] > puntos_equipos["equipo_2"]:
                self.ganadores_fases[fase] = "equipo_1"
                
                if fase == "PARES":
                    puntos_adicionales = self.calcular_puntos_pares("equipo_1")
                elif fase == "JUEGO":
                    puntos_adicionales = self.calcular_puntos_juego("equipo_1")
                else:
                    puntos_adicionales = 0
                    
                puntos_ganados = (self.apuesta_actual if self.apuesta_actual > 0 else 1) + puntos_adicionales
                self.puntos_equipos["equipo_1"] += puntos_ganados
                self.apuestas["equipo_1"][fase] = puntos_ganados
                
            elif puntos_equipos["equipo_2"] > puntos_equipos["equipo_1"]:
                self.ganadores_fases[fase] = "equipo_2"
                
                if fase == "PARES":
                    puntos_adicionales = self.calcular_puntos_pares("equipo_2")
                elif fase == "JUEGO":
                    puntos_adicionales = self.calcular_puntos_juego("equipo_2")
                else:
                    puntos_adicionales = 0
                    
                puntos_ganados = (self.apuesta_actual if self.apuesta_actual > 0 else 1) + puntos_adicionales
                self.puntos_equipos["equipo_2"] += puntos_ganados
                self.apuestas["equipo_2"][fase] = puntos_ganados
                
            else:
                self.ganadores_fases[fase] = None
                
        # Reiniciar la apuesta después de determinar el ganador
        self.apuesta_actual = 0
        self.equipo_apostador = None
        self.jugador_apostador = None
        self.hay_ordago = False

    def avanzar_fase(self):
        """Avanza a la siguiente fase del juego"""
        current_idx = self.fases.index(self.fase_actual)
        
        if current_idx < len(self.fases) - 1:
            next_idx = current_idx + 1
            next_fase = self.fases[next_idx]
            
            if next_fase == "PARES" and not any(self.declaraciones_pares.values()):
                if "JUEGO" in self.fases:
                    next_fase = "JUEGO"
                else:
                    next_fase = "RECUENTO"
            
            if next_fase == "JUEGO" and not any(self.declaraciones_juego.values()):
                # Solo saltar si no hay cartas (caso extremo)
                if all(not mano for mano in self.manos.values()):
                    next_fase = "RECUENTO"
                else:
                    # Permitir jugar al punto aunque nadie tenga juego
                    next_fase = "JUEGO"
                
            self.fase_actual = next_fase
            self.reiniciar_para_nueva_fase()
            
            if self.agent_selection not in self.jugadores_que_pueden_hablar:
                self.siguiente_jugador_que_puede_hablar()
                
            if self.fase_actual == "RECUENTO":
                self.determinar_ganador_global()
        else:
            self.fase_actual = "RECUENTO"
            for agent in self.agents:
                self.dones[agent] = True
            self.determinar_ganador_global()

    def determinar_ganador_global(self):
        """Determina el ganador de la partida actual y verifica si hay ganador del juego (mejor de 3)"""
        equipo_ganador_partida = max(self.puntos_equipos.items(), key=lambda x: x[1])[0]
        self.partidas_ganadas[equipo_ganador_partida] += 1
        
        # Verificar si algún equipo ganó 2 partidas
        if self.partidas_ganadas["equipo_1"] >= 2 or self.partidas_ganadas["equipo_2"] >= 2:
            ganador_juego = max(self.partidas_ganadas.items(), key=lambda x: x[1])[0]
            for agent in self.agents:
                if self.equipo_de_jugador[agent] == ganador_juego:
                    self.rewards[agent] = 100  # Recompensa alta por ganar el juego
                else:
                    self.rewards[agent] = -100
        else:
            # Si no hay ganador del juego, solo asignar recompensas por la partida
            for agent in self.agents:
                if self.equipo_de_jugador[agent] == equipo_ganador_partida:
                    self.rewards[agent] = 30
                else:
                    self.rewards[agent] = -30
        
        return equipo_ganador_partida

    def render(self):
        print(f"Fase: {self.fase_actual}")
        print(f"Jugadores que pueden hablar: {self.jugadores_que_pueden_hablar}")
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

    def calcular_puntos_pares(self, equipo):
        """Mejorar cálculo de puntos por pares"""
        puntos_totales = 0
        
        for jugador in self.equipos.get(equipo, []):
            if not self.declaraciones_pares.get(jugador, False):
                continue
                
            if jugador not in self.manos:
                continue
                
            mano = self.manos[jugador]
            valores = [carta[0] for carta in mano]
            conteo = {}
            for valor in valores:
                conteo[valor] = conteo.get(valor, 0) + 1
            
            if any(c == 4 for c in conteo.values()):
                puntos_totales += 3
            elif list(conteo.values()).count(2) >= 2:
                puntos_totales += 3
            elif any(c == 3 for c in conteo.values()):
                puntos_totales += 2
            elif any(c == 2 for c in conteo.values()):
                puntos_totales += 1
        
        return max(1, puntos_totales)

    def calcular_puntos_juego(self, equipo):
        """Mejorar cálculo de puntos por juego"""
        puntos_totales = 0
        
        for jugador in self.equipos.get(equipo, []):
            if not self.declaraciones_juego.get(jugador, False):
                continue
                
            if jugador not in self.valores_juego:
                continue
                
            valor_juego = self.valores_juego[jugador]
            
            if valor_juego == 31:
                puntos_totales += 3
            elif valor_juego == 32:
                puntos_totales += 2
            else:
                puntos_totales += 1
        
        return max(1, puntos_totales)

def env():
    return MusEnv()
