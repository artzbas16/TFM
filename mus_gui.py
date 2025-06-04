import random
import pygame
import sys
from mus_env import mus
import os
import time

class Boton:
    def __init__(self, x, y, texto, accion, ancho=150, alto=50):
        self.rect = pygame.Rect(x, y, ancho, alto)
        self.texto = texto
        self.accion = accion
        self.color_normal = (200, 200, 200)
        self.color_seleccionado = (150, 150, 255)
        self.color_actual = self.color_normal
        self.font = pygame.font.SysFont("Arial", 24)

    def dibujar(self, pantalla):
        pygame.draw.rect(pantalla, self.color_actual, self.rect, border_radius=12)
        pygame.draw.rect(pantalla, BLACK, self.rect, 2, border_radius=12)
        texto_render = self.font.render(self.texto, True, BLACK)
        texto_rect = texto_render.get_rect(center=self.rect.center)
        pantalla.blit(texto_render, texto_rect)

    def fue_click(self, pos):
        return self.rect.collidepoint(pos)

    def actualizar_estado(self, mouse_pos):
        self.color_actual = self.color_seleccionado if self.rect.collidepoint(mouse_pos) else self.color_normal

# Inicializa Pygame
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mus IA - 4 Reyes")
font = pygame.font.SysFont("Arial", 24)
font_small = pygame.font.SysFont("Arial", 18)
font_large = pygame.font.SysFont("Arial", 32, bold=True)
clock = pygame.time.Clock()

# Colores
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
LIGHT_GRAY = (220, 220, 220)
DARK_GREEN = (0, 100, 0)

acciones = {
    0: "Paso",
    1: "Envido",
    2: "Mus",
    3: "No Mus",
    4: "OK",
    5: "No quiero",
    6: "Órdago",
    7: "Quiero"
}

# Cargar entorno
mus_env = mus.env()
mus_env.reset()

# Tamaño y posiciones
agent_positions = [
    (WIDTH // 2, HEIGHT - 100),       # Jugador 0 (humano)
    (WIDTH - 200, HEIGHT // 2),       # Jugador 1
    (WIDTH // 2, 100),                # Jugador 2
    (200, HEIGHT // 2),               # Jugador 3
]

# Colores de los equipos
equipo_colors = {
    "equipo_1": BLUE,   # Equipo 1 (jugadores 0 y 2)
    "equipo_2": RED     # Equipo 2 (jugadores 1 y 3)
}

def cargar_cartas():
    """Carga las imágenes reales de las cartas desde la carpeta cartas"""
    palos = ['c', 'o', 'b', 'e']  # copas, oros, bastos, espadas
    cartas_img = {}
    
    try:
        path = os.path.join(os.path.dirname(__file__), "cartas")
        for palo_idx, palo in enumerate(palos):
            for num in range(1, 13):
                if num == 8 or num == 9:
                    continue
                nombre = f"{palo}{num}"
                archivo = os.path.join(path, f"{nombre}.png")
                try:
                    imagen = pygame.image.load(archivo)
                    imagen = pygame.transform.scale(imagen, (60, 100))
                    cartas_img[(num, palo_idx)] = imagen
                except pygame.error:
                    print(f"No se pudo cargar la imagen: {archivo}")
                    placeholder = pygame.Surface((60, 100))
                    placeholder.fill((255, 255, 255))
                    pygame.draw.rect(placeholder, BLACK, (0, 0, 60, 100), 2)
                    font_carta = pygame.font.SysFont("Arial", 18)
                    texto = font_carta.render(f"{palo}{num}", True, BLACK)
                    placeholder.blit(texto, (10, 40))
                    cartas_img[(num, palo_idx)] = placeholder
    except Exception as e:
        print(f"Error al cargar cartas: {e}")
        for palo_idx in range(4):
            for num in range(1, 13):
                if num == 8 or num == 9:
                    continue
                placeholder = pygame.Surface((60, 100))
                placeholder.fill((255, 255, 255))
                pygame.draw.rect(placeholder, BLACK, (0, 0, 60, 100), 2)
                font_carta = pygame.font.SysFont("Arial", 18)
                palos_nombres = ['C', 'O', 'B', 'E']
                texto = font_carta.render(f"{palos_nombres[palo_idx]}{num}", True, BLACK)
                placeholder.blit(texto, (10, 40))
                cartas_img[(num, palo_idx)] = placeholder
    
    return cartas_img

def cargar_reverso():
    """Carga la imagen del reverso de las cartas"""
    try:
        carta_reverso_path = os.path.join(os.path.dirname(__file__), "cartas", "rev.png")
        carta_reverso = pygame.image.load(carta_reverso_path)
        carta_reverso = pygame.transform.scale(carta_reverso, (60, 100))
        return carta_reverso
    except Exception as e:
        print(f"Error cargando reverso: {e}")
        reverso = pygame.Surface((60, 100))
        reverso.fill((50, 50, 150))
        pygame.draw.rect(reverso, (20, 20, 100), pygame.Rect(5, 5, 50, 90), 2)
        return reverso

def cargar_tapete():
    """Carga la imagen del tapete"""
    try:
        tapete_path = os.path.join(os.path.dirname(__file__), "cartas", "tapete.png")
        tapete = pygame.image.load(tapete_path)
        tapete = pygame.transform.scale(tapete, (WIDTH, HEIGHT))
        return tapete
    except Exception as e:
        print(f"Error cargando tapete: {e}")
        # Crear un fondo verde por defecto
        tapete = pygame.Surface((WIDTH, HEIGHT))
        tapete.fill(DARK_GREEN)
        return tapete
    
def draw_step(agent, accion):
    """Resalta el jugador actual y muestra su última decisión"""
    if agent not in mus_env.agents:
        return
        
    i = mus_env.agents.index(agent)
    x, y = agent_positions[i]
    
    # Solo dibujar si no está en fase de recuento y no está "done"
    if not mus_env.dones.get(agent, False) and mus_env.fase_actual != "RECUENTO":
        # Marco naranja más visible
        pygame.draw.rect(screen, ORANGE, (x - 140, y - 20, 310, 140), 4)
        pygame.draw.rect(screen, YELLOW, (x - 135, y - 15, 300, 130), 2)
        
        # Mostrar decisión del jugador actual
        decision = acciones.get(accion, "Desconocida")
        decision_texto = font_small.render(f"Decisión: {decision}", True, ORANGE)
        
        # Posicionamiento según la posición del jugador
        if i == 0:  # Jugador humano (abajo)
            screen.blit(decision_texto, (x - 120, y - 35))
        elif i == 1:  # Jugador derecha
            screen.blit(decision_texto, (x - 180, y - 70))
        elif i == 2:  # Jugador arriba
            screen.blit(decision_texto, (x - 120, y + 120))
        elif i == 3:  # Jugador izquierda
            screen.blit(decision_texto, (x + 20, y - 70))

def draw_table():
    # Dibujar fondo de la mesa
    screen.blit(tapete_fondo, (0, 0))

    if mus_env.partidas_ganadas["equipo_1"] >= 2 or mus_env.partidas_ganadas["equipo_2"] >= 2:
        draw_final_final_screen()
        return
    
    # En fase de recuento, mostrar todas las cartas y tabla centrada
    if mus_env.fase_actual == "RECUENTO":
        draw_final_screen()
        return
    
    partidas_texto = font.render(
        f"Partidas: Equipo 1 ({mus_env.partidas_ganadas['equipo_1']}) - Equipo 2 ({mus_env.partidas_ganadas['equipo_2']})", 
        True, WHITE
    )
    screen.blit(partidas_texto, (WIDTH // 2 - 150, 10))
    
    # Texto informativo
    fase_texto = font.render(f"Fase: {mus_env.fase_actual}", True, WHITE)
    screen.blit(fase_texto, (20, 10))
    
    turno_texto = font.render(f"Turno de: {mus_env.agent_selection}", True, WHITE)
    screen.blit(turno_texto, (20, 40))
    
    # Mostrar tabla de apuestas
    # Rectángulo de fondo
    pygame.draw.rect(screen, (40, 40, 40), (15, 85, 150, 160))
    pygame.draw.rect(screen, WHITE, (20, 90, 140, 150), 2)

    # Líneas horizontales ajustadas al nuevo ancho
    pygame.draw.line(screen, WHITE, (20, 120), (160, 120), 1)
    pygame.draw.line(screen, WHITE, (20, 150), (160, 150), 1)
    pygame.draw.line(screen, WHITE, (20, 180), (160, 180), 1)
    pygame.draw.line(screen, WHITE, (20, 210), (160, 210), 1)

    # Encabezado
    header = font_small.render("Apuestas", True, YELLOW)
    screen.blit(header, (50, 95))

    # Filas de datos
    fases = ["Grande", "Chica", "Pares", "Juego"]
    for i, fase in enumerate(fases):
        # Nombre de la fase
        fase_text = font_small.render(fase, True, WHITE)
        screen.blit(fase_text, (30, 125 + i * 30))
        
        # Puntos (suma de ambos equipos)
        puntos_eq1 = mus_env.apuestas["equipo_1"][fase.upper()] if hasattr(mus_env, 'apuestas') and "equipo_1" in mus_env.apuestas and fase.upper() in mus_env.apuestas["equipo_1"] else 0
        puntos_eq2 = mus_env.apuestas["equipo_2"][fase.upper()] if hasattr(mus_env, 'apuestas') and "equipo_2" in mus_env.apuestas and fase.upper() in mus_env.apuestas["equipo_2"] else 0
        puntos_text = font_small.render(str(puntos_eq1 + puntos_eq2), True, YELLOW)
        screen.blit(puntos_text, (100, 125 + i * 30))
        
    # Mostrar apuesta actual si hay una
    if mus_env.apuesta_actual > 0:
        if mus_env.equipo_apostador:
            apostador_texto = font.render(str(mus_env.apuesta_actual), True, equipo_colors[mus_env.equipo_apostador])
            screen.blit(apostador_texto, (130, 90))
            
        if hasattr(mus_env, 'hay_ordago') and mus_env.hay_ordago:
            ordago_texto = font.render("¡ÓRDAGO EN JUEGO!", True, RED)
            screen.blit(ordago_texto, (50, 230))

    # Dibujar cartas de los jugadores y marcar al jugador actual
    for i, agent in enumerate(mus_env.agents):
        x, y = agent_positions[i]
        # CORRECCIÓN: Dibujar un marco alrededor del jugador actual con mejor lógica
        
        
        # Mostrar el equipo al que pertenece cada jugador
        equipo = mus_env.equipo_de_jugador[agent]
        equipo_texto = font_large.render(f"{equipo}", True, equipo_colors[equipo])
        if i == 0:  # Jugador humano (abajo)
            screen.blit(equipo_texto, (x - 50, y - 70))
        elif i == 1:  # Jugador derecha
            screen.blit(equipo_texto, (x - 50, y - 70))
        elif i == 2:  # Jugador arriba
            screen.blit(equipo_texto, (x - 50, y - 70))
        elif i == 3:  # Jugador izquierda
            screen.blit(equipo_texto, (x - 50, y - 70))
        
        # Mostrar declaraciones SOLO en las fases correspondientes
        if mus_env.fase_actual == "PARES" and agent in mus_env.declaraciones_pares:
            tiene_pares = mus_env.declaraciones_pares[agent]
            pares_texto = font_small.render(f"{'Pares: Sí' if tiene_pares else 'Pares: No'}", True, YELLOW)
            if i == 0:  # Jugador humano (abajo)
                screen.blit(pares_texto, (x - 120, y - 20))
            elif i == 1:  # Jugador derecha
                screen.blit(pares_texto, (x - 180, y - 40))
            elif i == 2:  # Jugador arriba
                screen.blit(pares_texto, (x - 120, y + 110))
            elif i == 3:  # Jugador izquierda
                screen.blit(pares_texto, (x + 20, y - 40))
        
        if mus_env.fase_actual == "JUEGO" and agent in mus_env.declaraciones_juego:
            tiene_juego = mus_env.declaraciones_juego[agent]
            valor_juego = mus_env.valores_juego[agent]
            juego_texto = font_small.render(f"{'Juego: ' + str(valor_juego) if tiene_juego else 'Juego: No'}", True, YELLOW)
            if i == 0:  # Jugador humano (abajo)
                screen.blit(juego_texto, (x + 50, y - 20))
            elif i == 1:  # Jugador derecha
                screen.blit(juego_texto, (x - 180, y - 20))
            elif i == 2:  # Jugador arriba
                screen.blit(juego_texto, (x + 50, y + 110))
            elif i == 3:  # Jugador izquierda
                screen.blit(juego_texto, (x + 20, y - 20))
        
        # Mostrar si el jugador puede participar en la fase actual
        if mus_env.fase_actual in ["PARES", "JUEGO"]:
            puede_participar = agent in mus_env.jugadores_que_pueden_hablar
            participacion_texto = font_small.render(f"{'Puede jugar' if puede_participar else 'No puede jugar'}", True, GREEN if puede_participar else RED)
            if i == 0:  # Jugador humano (abajo)
                screen.blit(participacion_texto, (x - 120, y + 80))
            elif i == 1:  # Jugador derecha
                screen.blit(participacion_texto, (x - 180, y + 80))
            elif i == 2:  # Jugador arriba
                screen.blit(participacion_texto, (x - 120, y - 40))
            elif i == 3:  # Jugador izquierda
                screen.blit(participacion_texto, (x + 20, y + 80))
        
        # Mostrar cartas según la fase
        if i == 0:  # Jugador humano - siempre mostrar sus cartas
            mano = mus_env.manos[agent]
            for j, (valor, palo) in enumerate(mano):
                img = cartas_img.get((valor, palo))
                if img:
                    screen.blit(img, (x - 120 + j * 70, y))
                    if j in mus_env.cartas_a_descartar.get(agent, []):
                        pygame.draw.rect(screen, RED, (x - 120 + j * 70, y, 60, 100), 3)
        else:  # Otros jugadores - mostrar reverso
            for j in range(4):
                screen.blit(carta_reverso, (x - 120 + j * 70, y))
    
    # Dibujar botones según la fase actual y el contexto
    for boton in botones:
        if boton.accion in botones_visibles(mus_env.fase_actual, mus_env.agent_selection) or boton.accion == -1:
            boton.dibujar(screen)
    
    jugador_humano = "jugador_0"
    
    # Instrucciones según la fase actual
    if mus_env.fase_actual == "DESCARTE" and mus_env.agent_selection == jugador_humano:
        instrucciones = font.render("Selecciona cartas para descartar y pulsa OK", True, WHITE)
        screen.blit(instrucciones, (WIDTH // 2 - 200, HEIGHT // 2))
    
    # Mostrar si el jugador humano no puede participar en la fase actual
    if mus_env.fase_actual in ["PARES", "JUEGO"] and jugador_humano not in mus_env.jugadores_que_pueden_hablar:
        no_puede_texto = font.render(f"No puedes participar en {mus_env.fase_actual}", True, RED)
        screen.blit(no_puede_texto, (WIDTH // 2 - 150, HEIGHT // 2))

    # Mostrar puntos en fases de apuestas
    if mus_env.agent_selection == jugador_humano and mus_env.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
        if jugador_humano in mus_env.jugadores_que_pueden_hablar:

            if mus_env.fase_actual == "JUEGO":
                valor_juego = mus_env.valores_juego[jugador_humano]
                texto_valor = font.render(f"Valor de tu mano: {valor_juego}", True, WHITE)
                screen.blit(texto_valor, (WIDTH // 2 - 100, HEIGHT - 180))


def draw_final_final_screen():
    """Dibuja la pantalla final con todas las cartas visibles y la tabla de puntos centrada"""
    # Fondo semi-transparente
    overlay = pygame.Surface((WIDTH, HEIGHT))
    screen.blit(tapete_fondo, (0, 0))
    
    # Título principal
    titulo = font_large.render("¡PARTIDA TERMINADA!", True, YELLOW)
    titulo_rect = titulo.get_rect(center=(WIDTH // 2, 50))
    screen.blit(titulo, titulo_rect)

    if mus_env.partidas_ganadas["equipo_1"] >= 2 or mus_env.partidas_ganadas["equipo_2"] >= 2:
        ganador_juego = max(mus_env.partidas_ganadas.items(), key=lambda x: x[1])[0]
        texto_ganador = font_large.render(
            f"¡GANADOR DEL JUEGO: {ganador_juego.upper()}!", 
            True, equipo_colors[ganador_juego]
        )
        screen.blit(texto_ganador, (WIDTH // 2 - 200, HEIGHT - 100))
    
    # Mostrar todas las cartas de todos los jugadores
    for i, agent in enumerate(mus_env.agents):
        x, y = agent_positions[i]
        
        # Nombre del jugador y equipo
        equipo = mus_env.equipo_de_jugador[agent]
        nombre_texto = font.render(f"{agent} ({equipo})", True, equipo_colors[equipo])
        
        if i == 0:  # Jugador humano (abajo)
            screen.blit(nombre_texto, (x - 120, y - 30))
        elif i == 1:  # Jugador derecha
            screen.blit(nombre_texto, (x - 180, y - 30))
        elif i == 2:  # Jugador arriba
            screen.blit(nombre_texto, (x - 120, y + 110))
        elif i == 3:  # Jugador izquierda
            screen.blit(nombre_texto, (x + 20, y - 30))
        
        # Mostrar todas las cartas boca arriba
        if agent in mus_env.manos:
            mano = mus_env.manos[agent]
            for j, (valor, palo) in enumerate(mano):
                img = cartas_img.get((valor, palo))
                if img:
                    screen.blit(img, (x - 120 + j * 70, y))
    
    # Tabla de puntos centrada
    tabla_x = WIDTH // 2 - 250
    tabla_y = HEIGHT // 2 - 100
    tabla_ancho = 500
    tabla_alto = 200
    
    # Fondo de la tabla
    pygame.draw.rect(screen, (40, 40, 40), (tabla_x - 10, tabla_y - 10, tabla_ancho + 20, tabla_alto + 20))
    pygame.draw.rect(screen, WHITE, (tabla_x, tabla_y, tabla_ancho, tabla_alto), 3)
    
    # Título de la tabla
    titulo_tabla = font_large.render("PUNTUACIÓN FINAL", True, YELLOW)
    titulo_rect = titulo_tabla.get_rect(center=(WIDTH // 2, tabla_y - 30))
    screen.blit(titulo_tabla, titulo_rect)
    
    # Encabezados
    header_y = tabla_y + 20
    pygame.draw.line(screen, WHITE, (tabla_x, header_y + 30), (tabla_x + tabla_ancho, header_y + 30), 2)
    
    encabezados = ["FASE", "EQUIPO 1", "EQUIPO 2"]
    col_width = tabla_ancho // 3
    
    for i, encabezado in enumerate(encabezados):
        texto = font.render(encabezado, True, WHITE)
        texto_rect = texto.get_rect(center=(tabla_x + col_width * i + col_width // 2, header_y + 15))
        screen.blit(texto, texto_rect)
    
    # Filas de datos
    fases = ["GRANDE", "CHICA", "PARES", "JUEGO"]
    for row, fase in enumerate(fases):
        row_y = header_y + 50 + row * 30
        
        # Nombre de la fase
        fase_texto = font.render(fase, True, WHITE)
        fase_rect = fase_texto.get_rect(center=(tabla_x + col_width // 2, row_y))
        screen.blit(fase_texto, fase_rect)
        
        # Puntos equipo 1
        puntos_eq1 = mus_env.apuestas["equipo_1"][fase] if hasattr(mus_env, 'apuestas') and "equipo_1" in mus_env.apuestas else 0
        eq1_texto = font.render(str(puntos_eq1), True, equipo_colors["equipo_1"])
        eq1_rect = eq1_texto.get_rect(center=(tabla_x + col_width + col_width // 2, row_y))
        screen.blit(eq1_texto, eq1_rect)
        
        # Puntos equipo 2
        puntos_eq2 = mus_env.apuestas["equipo_2"][fase] if hasattr(mus_env, 'apuestas') and "equipo_2" in mus_env.apuestas else 0
        eq2_texto = font.render(str(puntos_eq2), True, equipo_colors["equipo_2"])
        eq2_rect = eq2_texto.get_rect(center=(tabla_x + col_width * 2 + col_width // 2, row_y))
        screen.blit(eq2_texto, eq2_rect)
    
    # Línea de separación para totales
    total_y = header_y + 50 + len(fases) * 30
    pygame.draw.line(screen, WHITE, (tabla_x, total_y), (tabla_x + tabla_ancho, total_y), 2)
    
    # Totales
    total_eq1 = mus_env.puntos_equipos["equipo_1"]
    total_eq2 = mus_env.puntos_equipos["equipo_2"]
    
    total_texto = font_large.render("TOTAL", True, YELLOW)
    total_rect = total_texto.get_rect(center=(tabla_x + col_width // 2, total_y + 25))
    screen.blit(total_texto, total_rect)
    
    total1_texto = font_large.render(str(total_eq1), True, equipo_colors["equipo_1"])
    total1_rect = total1_texto.get_rect(center=(tabla_x + col_width + col_width // 2, total_y + 25))
    screen.blit(total1_texto, total1_rect)
    
    total2_texto = font_large.render(str(total_eq2), True, equipo_colors["equipo_2"])
    total2_rect = total2_texto.get_rect(center=(tabla_x + col_width * 2 + col_width // 2, total_y + 25))
    screen.blit(total2_texto, total2_rect)
    
    # Ganador
    ganador_y = total_y + 70
    if total_eq1 > total_eq2:
        ganador_texto = font_large.render("¡GANADOR: EQUIPO 1 (Jugadores 0 y 2)!", True, equipo_colors["equipo_1"])
    elif total_eq2 > total_eq1:
        ganador_texto = font_large.render("¡GANADOR: EQUIPO 2 (Jugadores 1 y 3)!", True, equipo_colors["equipo_2"])
    else:
        ganador_texto = font_large.render("¡EMPATE!", True, YELLOW)
    
    ganador_rect = ganador_texto.get_rect(center=(WIDTH // 2, ganador_y))
    screen.blit(ganador_texto, ganador_rect)
    
    # Instrucciones
    instruccion = font.render("Presiona ESC para salir o ESPACIO para nueva partida", True, WHITE)
    instruccion_rect = instruccion.get_rect(center=(WIDTH // 2, HEIGHT - 150))
    screen.blit(instruccion, instruccion_rect)

def draw_final_screen():
    """Dibuja la pantalla final con todas las cartas visibles y la tabla de puntos centrada"""
    # Fondo semi-transparente
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((0, 50, 0))
    screen.blit(overlay, (0, 0))
    
    # Título principal
    titulo = font_large.render("¡RONDA TERMINADA!", True, YELLOW)
    titulo_rect = titulo.get_rect(center=(WIDTH // 2, 50))
    screen.blit(titulo, titulo_rect)
    
    # Mostrar todas las cartas de todos los jugadores
    for i, agent in enumerate(mus_env.agents):
        x, y = agent_positions[i]
        
        # Nombre del jugador y equipo
        equipo = mus_env.equipo_de_jugador[agent]
        nombre_texto = font.render(f"{agent} ({equipo})", True, equipo_colors[equipo])
        
        if i == 0:  # Jugador humano (abajo)
            screen.blit(nombre_texto, (x - 120, y - 30))
        elif i == 1:  # Jugador derecha
            screen.blit(nombre_texto, (x - 180, y - 30))
        elif i == 2:  # Jugador arriba
            screen.blit(nombre_texto, (x - 120, y + 110))
        elif i == 3:  # Jugador izquierda
            screen.blit(nombre_texto, (x + 20, y - 30))
        
        # Mostrar todas las cartas boca arriba
        if agent in mus_env.manos:
            mano = mus_env.manos[agent]
            for j, (valor, palo) in enumerate(mano):
                img = cartas_img.get((valor, palo))
                if img:
                    screen.blit(img, (x - 120 + j * 70, y))
    
    # Tabla de puntos centrada
    tabla_x = WIDTH // 2 - 250
    tabla_y = HEIGHT // 2 - 100
    tabla_ancho = 500
    tabla_alto = 200
    
    # Fondo de la tabla
    pygame.draw.rect(screen, (40, 40, 40), (tabla_x - 10, tabla_y - 10, tabla_ancho + 20, tabla_alto + 20))
    pygame.draw.rect(screen, WHITE, (tabla_x, tabla_y, tabla_ancho, tabla_alto), 3)
    
    # Título de la tabla
    titulo_tabla = font_large.render("PUNTUACIÓN", True, YELLOW)
    titulo_rect = titulo_tabla.get_rect(center=(WIDTH // 2, tabla_y - 30))
    screen.blit(titulo_tabla, titulo_rect)
    
    # Encabezados
    header_y = tabla_y + 20
    pygame.draw.line(screen, WHITE, (tabla_x, header_y + 30), (tabla_x + tabla_ancho, header_y + 30), 2)
    
    encabezados = ["FASE", "EQUIPO 1", "EQUIPO 2"]
    col_width = tabla_ancho // 3
    
    for i, encabezado in enumerate(encabezados):
        texto = font.render(encabezado, True, WHITE)
        texto_rect = texto.get_rect(center=(tabla_x + col_width * i + col_width // 2, header_y + 15))
        screen.blit(texto, texto_rect)
    
    # Filas de datos
    fases = ["GRANDE", "CHICA", "PARES", "JUEGO"]
    for row, fase in enumerate(fases):
        row_y = header_y + 50 + row * 30
        
        # Nombre de la fase
        fase_texto = font.render(fase, True, WHITE)
        fase_rect = fase_texto.get_rect(center=(tabla_x + col_width // 2, row_y))
        screen.blit(fase_texto, fase_rect)
        
        # Puntos equipo 1
        puntos_eq1 = mus_env.apuestas["equipo_1"][fase] if hasattr(mus_env, 'apuestas') and "equipo_1" in mus_env.apuestas else 0
        eq1_texto = font.render(str(puntos_eq1), True, equipo_colors["equipo_1"])
        eq1_rect = eq1_texto.get_rect(center=(tabla_x + col_width + col_width // 2, row_y))
        screen.blit(eq1_texto, eq1_rect)
        
        # Puntos equipo 2
        puntos_eq2 = mus_env.apuestas["equipo_2"][fase] if hasattr(mus_env, 'apuestas') and "equipo_2" in mus_env.apuestas else 0
        eq2_texto = font.render(str(puntos_eq2), True, equipo_colors["equipo_2"])
        eq2_rect = eq2_texto.get_rect(center=(tabla_x + col_width * 2 + col_width // 2, row_y))
        screen.blit(eq2_texto, eq2_rect)
    
    # Línea de separación para totales
    total_y = header_y + 50 + len(fases) * 30
    pygame.draw.line(screen, WHITE, (tabla_x, total_y), (tabla_x + tabla_ancho, total_y), 2)
    
    # Totales
    total_eq1 = mus_env.puntos_equipos["equipo_1"]
    total_eq2 = mus_env.puntos_equipos["equipo_2"]
    
    total_texto = font_large.render("TOTAL", True, YELLOW)
    total_rect = total_texto.get_rect(center=(tabla_x + col_width // 2, total_y + 25))
    screen.blit(total_texto, total_rect)
    
    total1_texto = font_large.render(str(total_eq1), True, equipo_colors["equipo_1"])
    total1_rect = total1_texto.get_rect(center=(tabla_x + col_width + col_width // 2, total_y + 25))
    screen.blit(total1_texto, total1_rect)
    
    total2_texto = font_large.render(str(total_eq2), True, equipo_colors["equipo_2"])
    total2_rect = total2_texto.get_rect(center=(tabla_x + col_width * 2 + col_width // 2, total_y + 25))
    screen.blit(total2_texto, total2_rect)
    
    # Instrucciones
    instruccion = font.render("Presiona ESC para salir o ESPACIO para siguiente ronda", True, WHITE)
    instruccion_rect = instruccion.get_rect(center=(WIDTH // 2, HEIGHT - 150))
    screen.blit(instruccion, instruccion_rect)

def botones_visibles(fase_actual, jugador_actual):
    """Determina qué botones deben estar visibles según la fase y el contexto"""
    if fase_actual == "RECUENTO":
        return []  # No mostrar botones en la fase final
    elif fase_actual == "MUS":
        return [2, 3]  # Mus / No Mus
    elif fase_actual == "DESCARTE":
        return [4]  # Solo botón OK para confirmar
    elif fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
        if jugador_actual not in mus_env.jugadores_que_pueden_hablar:
            return []
            
        if hasattr(mus_env, 'hay_ordago') and mus_env.hay_ordago:
            if mus_env.equipo_apostador and mus_env.equipo_de_jugador[jugador_actual] != mus_env.equipo_apostador:
                return [5, 7]  # No quiero, Quiero
            else:
                return []
        elif mus_env.apuesta_actual > 0:
            if mus_env.equipo_apostador and mus_env.equipo_de_jugador[jugador_actual] != mus_env.equipo_apostador:
                return [1, 5, 6, 7]  # Envido (subir), No quiero, Órdago, Quiero
            else:
                return [0, 1, 6]  # Paso, Envido, Órdago
        else:
            return [0, 1, 6]  # Paso, Envido, Órdago
    
    return []

def main():
    running = True
    global cartas_img, carta_reverso, botones, tapete_fondo
    
    # Cargar imágenes
    cartas_img = cargar_cartas()
    carta_reverso = cargar_reverso()
    tapete_fondo = cargar_tapete()

    # Botones para todas las acciones posibles
    botones = [
        Boton(680, 550, "Paso", 0),
        Boton(830, 550, "Envido", 1),
        Boton(750, 550, "Mus", 2),
        Boton(750, 600, "No Mus", 3),
        Boton(750, 550, "OK", 4),
        Boton(680, 600, "No quiero", 5),
        Boton(830, 600, "Órdago", 6),
        Boton(680, 550, "Quiero", 7),
        Boton(WIDTH - 180, 50, "Salir", -1)
    ]

    mouse_pos = (0, 0)
    jugador_humano = "jugador_0"

    while running:
        # Lógica de IA con delay automático (solo si no es fase de recuento)
        current_agent = mus_env.agent_selection
        if current_agent != jugador_humano and mus_env.fase_actual != "RECUENTO" and not mus_env.dones.get(current_agent, False):
            # Verificar si el jugador puede hablar en esta fase
            if mus_env.fase_actual in ["PARES", "JUEGO"] and current_agent not in mus_env.jugadores_que_pueden_hablar:
                mus_env.siguiente_jugador_que_puede_hablar()
                continue
            
            if mus_env.fase_actual == "MUS":
                # IA decide entre Mus y No Mus
                if random.random() > 0.8:  # 20% probabilidad de decir "No Mus"
                    action = 3  # No Mus
                else:
                    action = 2  # Mus

            elif mus_env.fase_actual == "DESCARTE":
                if current_agent not in mus_env.cartas_a_descartar or not mus_env.cartas_a_descartar[current_agent]:
                    num_descartes = random.randint(0, 2)
                    descartes = random.sample(range(4), num_descartes) if num_descartes > 0 else []
                    mus_env.cartas_a_descartar[current_agent] = descartes
                action = 4

            elif mus_env.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
                puntos = mus_env.calcular_puntos(mus_env.manos[current_agent], mus_env.fase_actual)
                
                # Lógica de IA mejorada para las fases de apuestas
                if hasattr(mus_env, 'hay_ordago') and mus_env.hay_ordago:
                    if mus_env.equipo_apostador and mus_env.equipo_de_jugador[current_agent] != mus_env.equipo_apostador:
                        action = 7  # Quiero (capear)
                    else:
                        action = 0  # Paso
                elif mus_env.apuesta_actual > 0:
                    if mus_env.equipo_apostador and mus_env.equipo_de_jugador[current_agent] != mus_env.equipo_apostador:
                        action = 1  # Envido (subir)
                    else:
                        action = 0  # Paso
                else:
                    
                    action = 0  # Paso
            
            draw_step(current_agent, action)
            pygame.display.flip() 
            mus_env.step(action)

        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and mus_env.fase_actual == "RECUENTO":
                    mus_env.reset()
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Manejar botón Salir
                for boton in botones:
                    if boton.fue_click(mouse_pos) and boton.accion == -1:
                        running = False
                        break
                
                # No procesar clics en fase de recuento
                if mus_env.fase_actual == "RECUENTO":
                    continue
                
                # Verificar si es turno del jugador humano
                if mus_env.agent_selection == "jugador_0":
                    if mus_env.fase_actual in ["PARES", "JUEGO"] and jugador_humano not in mus_env.jugadores_que_pueden_hablar:
                        # Si no puede hablar, pasar al siguiente jugador automáticamente
                        mus_env.siguiente_jugador_que_puede_hablar()
                        continue
                        
                    # Manejar clic en botones
                    boton_pulsado = None
                    for boton in botones:
                        if boton.fue_click(mouse_pos) and boton.accion in botones_visibles(mus_env.fase_actual, mus_env.agent_selection):
                            boton_pulsado = boton
                            break
                    
                    if boton_pulsado:
                        mus_env.step(boton_pulsado.accion)
                    elif mus_env.fase_actual == "DESCARTE":
                        # Manejar clic en cartas
                        x, y = agent_positions[0]
                        for j in range(4):
                            carta_rect = pygame.Rect(x - 120 + j * 70, y, 60, 100)
                            if carta_rect.collidepoint(mouse_pos):
                                mus_env.step(11 + j)
                                break
        # Actualizar estado visual de los botones (hover)
        for boton in botones:
            boton.actualizar_estado(mouse_pos)
        
        draw_table()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
