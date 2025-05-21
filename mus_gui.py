import random
import pygame
import sys
from mus_env import mus
import os

class Boton:
    def __init__(self, x, y, texto, accion, ancho=100, alto=40):
        self.rect = pygame.Rect(x, y, ancho, alto)
        self.texto = texto
        self.accion = accion
        self.color_normal = (200, 200, 200)
        self.color_seleccionado = (150, 150, 255)
        self.color_actual = self.color_normal
        self.font = pygame.font.SysFont("Arial", 24)

    def dibujar(self, pantalla):
        pygame.draw.rect(pantalla, self.color_actual, self.rect)
        pygame.draw.rect(pantalla, BLACK, self.rect, 2)
        texto_render = self.font.render(self.texto, True, BLACK)
        # Centrar el texto en el botón
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

# Cargar entorno
env = mus.env()
env.reset()

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
    palos = ['c', 'e', 'o', 'b']
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
                    # Create a placeholder surface
                    placeholder = pygame.Surface((60, 100))
                    placeholder.fill((255, 255, 255))
                    cartas_img[(num, palo_idx)] = placeholder
    except Exception as e:
        print(f"Error al cargar cartas: {e}")
    return cartas_img

def draw_table():
    screen.fill(GREEN)
    pygame.draw.circle(screen, WHITE, (WIDTH // 2, HEIGHT // 2), 200, 5)
    
    # Texto informativo
    fase_texto = font.render(f"Fase: {env.fase_actual}", True, WHITE)
    screen.blit(fase_texto, (20, 10))
    
    turno_texto = font.render(f"Turno de: {env.agent_selection}", True, WHITE)
    screen.blit(turno_texto, (20, 40))
    
    # Mostrar puntos de los equipos
    equipo1_texto = font.render(f"Equipo 1 (0,2): {env.puntos_equipos['equipo_1']}", True, equipo_colors["equipo_1"])
    screen.blit(equipo1_texto, (20, 70))
    
    equipo2_texto = font.render(f"Equipo 2 (1,3): {env.puntos_equipos['equipo_2']}", True, equipo_colors["equipo_2"])
    screen.blit(equipo2_texto, (20, 100))
    
    # Mostrar apuesta actual si hay una
    if env.apuesta_actual > 0:
        apuesta_texto = font.render(f"Apuesta actual: {env.apuesta_actual}", True, YELLOW)
        screen.blit(apuesta_texto, (20, 130))
        
        if env.equipo_apostador:
            apostador_texto = font.render(f"Equipo apostador: {env.equipo_apostador}", True, equipo_colors[env.equipo_apostador])
            screen.blit(apostador_texto, (20, 160))

    # Dibujar cartas de los jugadores y marcar al jugador actual
    for i, agent in enumerate(env.agents):
        x, y = agent_positions[i]
        
        # Dibujar un marco alrededor del jugador actual
        if agent == env.agent_selection:
            # Marco más grande para el jugador actual
            pygame.draw.rect(screen, ORANGE, (x - 130, y - 10, 290, 120), 3)
            
            # Mostrar la última decisión del jugador
            decision_texto = font_small.render(f"Decisión: {env.ultima_decision[agent]}", True, ORANGE)
            if i == 0:  # Jugador humano (abajo)
                screen.blit(decision_texto, (x - 120, y - 30))
            elif i == 1:  # Jugador derecha
                screen.blit(decision_texto, (x - 180, y - 60))
            elif i == 2:  # Jugador arriba
                screen.blit(decision_texto, (x - 120, y + 110))
            elif i == 3:  # Jugador izquierda
                screen.blit(decision_texto, (x + 20, y - 60))
        
        # Mostrar el equipo al que pertenece cada jugador
        equipo = env.equipo_de_jugador[agent]
        equipo_texto = font_small.render(f"{equipo}", True, equipo_colors[equipo])
        if i == 0:  # Jugador humano (abajo)
            screen.blit(equipo_texto, (x + 100, y + 50))
        elif i == 1:  # Jugador derecha
            screen.blit(equipo_texto, (x - 50, y + 50))
        elif i == 2:  # Jugador arriba
            screen.blit(equipo_texto, (x - 100, y - 20))
        elif i == 3:  # Jugador izquierda
            screen.blit(equipo_texto, (x + 50, y - 20))
        
        # Mostrar declaraciones SOLO en las fases correspondientes
        # Declaraciones de pares solo en fase PARES
        if env.fase_actual == "PARES" and agent in env.declaraciones_pares:
            tiene_pares = env.declaraciones_pares[agent]
            pares_texto = font_small.render(f"{'Pares: Sí' if tiene_pares else 'Pares: No'}", True, YELLOW)
            if i == 0:  # Jugador humano (abajo)
                screen.blit(pares_texto, (x - 120, y - 20))
            elif i == 1:  # Jugador derecha
                screen.blit(pares_texto, (x - 180, y - 40))
            elif i == 2:  # Jugador arriba
                screen.blit(pares_texto, (x - 120, y + 110))
            elif i == 3:  # Jugador izquierda
                screen.blit(pares_texto, (x + 20, y - 40))
        
        # Declaraciones de juego solo en fase JUEGO
        if env.fase_actual == "JUEGO" and agent in env.declaraciones_juego:
            tiene_juego = env.declaraciones_juego[agent]
            valor_juego = env.valores_juego[agent]
            juego_texto = font_small.render(f"{'Juego: ' + str(valor_juego) if tiene_juego else 'Juego: No'}", True, YELLOW)
            if i == 0:  # Jugador humano (abajo)
                screen.blit(juego_texto, (x + 50, y - 20))
            elif i == 1:  # Jugador derecha
                screen.blit(juego_texto, (x - 180, y - 20))
            elif i == 2:  # Jugador arriba
                screen.blit(juego_texto, (x + 50, y + 110))
            elif i == 3:  # Jugador izquierda
                screen.blit(juego_texto, (x + 20, y - 20))
        
        # En fase de recuento, mostrar todas las cartas
        if env.fase_actual == "RECUENTO":
            mano = env.manos[agent]
            for j, (valor, palo) in enumerate(mano):
                img = cartas_img.get((valor, palo))
                if img:
                    if i == 0:  # Jugador humano (abajo)
                        screen.blit(img, (x - 120 + j * 70, y))
                    elif i == 1:  # Jugador derecha
                        screen.blit(img, (x - 50, y - 120 + j * 30))
                    elif i == 2:  # Jugador arriba
                        screen.blit(img, (x - 120 + j * 70, y))
                    elif i == 3:  # Jugador izquierda
                        screen.blit(img, (x - 50, y - 120 + j * 30))
        else:
            # En otras fases, mostrar solo las cartas del jugador humano
            if i == 0:  # Jugador humano
                mano = env.manos[agent]
                for j, (valor, palo) in enumerate(mano):
                    img = cartas_img.get((valor, palo))
                    if img:
                        screen.blit(img, (x - 120 + j * 70, y))
                        if j in env.cartas_a_descartar.get(agent, []):
                            pygame.draw.rect(screen, RED, (x - 120 + j * 70, y, 60, 100), 3)
            else:  # Otros jugadores
                for j in range(4):
                    screen.blit(carta_reverso, (x - 120 + j * 70, y))
    
    # Dibujar botones según la fase actual y el contexto
    for boton in botones:
        if boton.accion in botones_visibles(env.fase_actual, env.agent_selection) or boton.accion == -1:  # -1 es siempre visible (Salir)
            boton.dibujar(screen)
    
    jugador_humano = "jugador_0"
    
    # Instrucciones según la fase actual
    if env.fase_actual == "DESCARTE" and env.agent_selection == jugador_humano:
        instrucciones = font.render("Selecciona cartas para descartar y pulsa OK", True, WHITE)
        screen.blit(instrucciones, (WIDTH // 2 - 200, HEIGHT // 2))

    # Mostrar puntos en fases de apuestas
    if env.agent_selection == jugador_humano and env.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
        puntos = env.calcular_puntos(env.manos[jugador_humano], env.fase_actual)
        texto_puntos = font.render(f"Tus puntos: {puntos}", True, WHITE)
        screen.blit(texto_puntos, (WIDTH // 2 - 100, HEIGHT - 150))
        
        # Mostrar valor de la mano en fase de JUEGO
        if env.fase_actual == "JUEGO":
            valor_juego = env.valores_juego[jugador_humano]
            texto_valor = font.render(f"Valor de tu mano: {valor_juego}", True, BLUE)
            screen.blit(texto_valor, (WIDTH // 2 - 100, HEIGHT - 180))

    # Mostrar ganadores de cada fase
    y_pos = 190
    for fase, ganador in env.ganadores_fases.items():
        if ganador:
            texto = font.render(
                f"{fase}: Ganador -> {ganador}", 
                True, equipo_colors[ganador]
            )
            screen.blit(texto, (WIDTH - 350, y_pos))
            y_pos += 30
            
    # En fase de recuento, mostrar los puntos totales de cada equipo
    if env.fase_actual == "RECUENTO":
        recuento_texto = font.render("RECUENTO FINAL", True, YELLOW)
        screen.blit(recuento_texto, (WIDTH // 2 - 100, HEIGHT // 2 - 100))
        
        y_pos = HEIGHT // 2 - 70
        for equipo, puntos in env.puntos_equipos.items():
            texto = font.render(f"{equipo}: {puntos} puntos", True, equipo_colors[equipo])
            screen.blit(texto, (WIDTH // 2 - 100, y_pos))
            y_pos += 30
            
        # Mostrar equipo ganador
        equipo_ganador = max(env.puntos_equipos.items(), key=lambda x: x[1])[0]
        texto_ganador = font.render(f"¡GANADOR: {equipo_ganador}!", True, equipo_colors[equipo_ganador])
        screen.blit(texto_ganador, (WIDTH // 2 - 100, y_pos + 20))

def botones_visibles(fase_actual, jugador_actual):
    """Determina qué botones deben estar visibles según la fase y el contexto"""
    visibles = []
    
    if fase_actual == "MUS":
        return [2, 3]  # Mus / No Mus
    elif fase_actual == "DESCARTE":
        return [4]  # Solo botón OK para confirmar
    elif fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
        # Si hay una apuesta activa
        if env.apuesta_actual > 0:
            # Si el jugador es del equipo contrario al que apostó
            if env.equipo_apostador and env.equipo_de_jugador[jugador_actual] != env.equipo_apostador:
                return [1, 5, 6, 7]  # Envido, No quiero, Órdago, Quiero
            else:
                # Si es del mismo equipo que apostó
                return [0, 1, 6]  # Paso, Envido, Órdago
        else:
            # Si no hay apuesta activa
            return [0, 1, 6]  # Paso, Envido, Órdago
    
    return visibles

def main():
    running = True
    global cartas_img, carta_reverso, botones
    
    try:
        cartas_img = cargar_cartas()
        # Usar os.path.join para la ruta de la carta reverso
        carta_reverso_path = os.path.join(os.path.dirname(__file__), "cartas", "rev.png")
        carta_reverso = pygame.image.load(carta_reverso_path)
        carta_reverso = pygame.transform.scale(carta_reverso, (60, 100))
    except Exception as e:
        print(f"Error cargando imágenes de cartas: {e}")
        # Crear un reverso de carta por defecto si no se puede cargar
        carta_reverso = pygame.Surface((60, 100))
        carta_reverso.fill((50, 50, 150))
        pygame.draw.rect(carta_reverso, (20, 20, 100), pygame.Rect(5, 5, 50, 90), 2)

    # Botones para todas las acciones posibles
    botones = [
        Boton(50, 150, "Mus", 2),
        Boton(50, 200, "No Mus", 3),
        Boton(50, 250, "Envido", 1),
        Boton(50, 300, "Paso", 0),
        Boton(50, 350, "No quiero", 5),
        Boton(50, 400, "Órdago", 6),
        Boton(50, 450, "Quiero", 7),  # Nuevo botón para "Quiero" (capear)
        Boton(WIDTH - 150, 150, "OK", 4),
        Boton(WIDTH - 150, 200, "Salir", -1)
    ]

    mouse_pos = (0, 0)
    
    jugador_humano = "jugador_0"

    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Manejar botón Salir primero (siempre disponible)
                for boton in botones:
                    if boton.fue_click(mouse_pos) and boton.accion == -1:
                        running = False
                        break
                # Verificar si es turno del jugador humano
                if env.agent_selection == "jugador_0":
                    # Verificar si el jugador puede hablar en esta fase
                    if env.fase_actual in ["PARES", "JUEGO"] and not env.puede_hablar(env.agent_selection):
                        # Si no puede hablar, pasar al siguiente jugador automáticamente
                        env.siguiente_jugador_que_puede_hablar()
                        continue
                        
                    # Manejar clic en botones
                    boton_pulsado = None
                    for boton in botones:
                        if boton.fue_click(mouse_pos) and boton.accion in botones_visibles(env.fase_actual, env.agent_selection):
                            boton_pulsado = boton
                            break
                    
                    if boton_pulsado:
                        env.step(boton_pulsado.accion)
                    elif env.fase_actual == "DESCARTE":
                        # Manejar clic en cartas (ajustado por las nuevas acciones)
                        x, y = agent_positions[0]
                        for j in range(4):
                            carta_rect = pygame.Rect(x - 120 + j * 70, y, 60, 100)
                            if carta_rect.collidepoint(mouse_pos):
                                env.step(11 + j)  # Ajustado a 11-14 para descartar
                                break
        
        # Lógica de IA
        current_agent = env.agent_selection
        if current_agent != jugador_humano:
            pygame.time.delay(300)  # Pequeña pausa para ver el turno de la IA
            
            # Verificar si el jugador puede hablar en esta fase
            if env.fase_actual in ["PARES", "JUEGO"] and not env.puede_hablar(current_agent):
                # Si no puede hablar, pasar al siguiente jugador automáticamente
                env.siguiente_jugador_que_puede_hablar()
                continue
            
            if env.fase_actual == "MUS":
                action = 2  # IA siempre dice Mus
                env.step(action)

            elif env.fase_actual == "DESCARTE":
                if current_agent not in env.cartas_a_descartar or not env.cartas_a_descartar[current_agent]:
                    num_descartes = random.randint(0, 2)
                    descartes = random.sample(range(4), num_descartes) if num_descartes > 0 else []
                    env.cartas_a_descartar[current_agent] = descartes
                env.step(4)

            elif env.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
                puntos = env.calcular_puntos(env.manos[current_agent], env.fase_actual)
                
                # Lógica de IA mejorada para las fases de apuestas
                if env.apuesta_actual > 0:
                    # Si hay una apuesta activa
                    if env.equipo_apostador and env.equipo_de_jugador[current_agent] != env.equipo_apostador:
                        # Si es del equipo contrario al que apostó
                        if puntos > 20:
                            if random.random() > 0.7:
                                action = 1  # Envido (subir)
                            else:
                                action = 7  # Quiero (capear)
                        else:
                            action = 5  # No quiero
                    else:
                        # Si es del mismo equipo que apostó
                        if puntos > 20 and random.random() > 0.7:
                            action = 1  # Envido (subir)
                        else:
                            action = 0  # Paso
                else:
                    # Si no hay apuesta activa
                    if puntos > 20 and random.random() > 0.5:
                        action = 1  # Envido
                    elif puntos > 25 and random.random() > 0.8:
                        action = 6  # Órdago
                    else:
                        action = 0  # Paso
                
                env.step(action)
        
        # Actualizar estado visual de los botones (hover)
        for boton in botones:
            boton.actualizar_estado(mouse_pos)
        
        draw_table()
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()