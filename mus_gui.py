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
clock = pygame.time.Clock()

# Colores
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

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

    # Dibujar cartas de los jugadores
    for i, agent in enumerate(env.agents):
        x, y = agent_positions[i]
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
    
    # Dibujar botones según la fase actual
    for boton in botones:
        if boton.accion in botones_visibles(env.fase_actual) or boton.accion == -1:  # -1 es siempre visible (Salir)
            boton.dibujar(screen)
    
    jugador_humano = "jugador_0"
    # Instrucciones durante el descarte
    if env.fase_actual == "DESCARTE" and env.agent_selection == jugador_humano:
        instrucciones = font.render("Selecciona cartas para descartar y pulsa OK", True, WHITE)
        screen.blit(instrucciones, (WIDTH // 2 - 200, HEIGHT // 2))

    if env.agent_selection == jugador_humano and env.fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
        puntos = env.calcular_puntos(env.manos[jugador_humano], env.fase_actual)
        texto_puntos = font.render(f"Tus puntos: {puntos}", True, WHITE)
        screen.blit(texto_puntos, (WIDTH // 2 - 100, HEIGHT - 150))

    y_pos = 70
    for fase in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
        if env.apuestas[fase]["ganador"]:
            texto = font.render(
                f"{fase}: {env.apuestas[fase]['valor']} -> {env.apuestas[fase]['ganador']}", 
                True, WHITE
            )
            screen.blit(texto, (WIDTH - 300, y_pos))
            y_pos += 30

def botones_visibles(fase_actual):
    visibles = []
    if fase_actual == "MUS":
        return [2, 3]  # Mus / No Mus
    elif fase_actual == "DESCARTE":
        return [4]  # Solo botón OK para confirmar
    elif fase_actual in ["GRANDE", "CHICA", "PARES", "JUEGO"]:
        return [0, 1, 5, 6]   # Pasar/Envite/No quiero/Ordago
    else:
        return visibles

def main():
    running = True
    global cartas_img, carta_reverso, botones
    
    try:
        cartas_img = cargar_cartas()
        carta_reverso = pygame.image.load("cartas/rev.png")
        carta_reverso = pygame.transform.scale(carta_reverso, (60, 100))
    except:
        print("Error cargando imágenes de cartas")
        running = False

    botones = [
        Boton(50, 150, "Mus", 2),
        Boton(50, 200, "No Mus", 3),
        Boton(50, 250, "Envido", 1),
        Boton(50, 300, "Paso", 0),
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
                    # Manejar clic en botones
                    boton_pulsado = None
                    for boton in botones:
                        if boton.fue_click(mouse_pos) and boton.accion in botones_visibles(env.fase_actual):
                            boton_pulsado = boton
                            break
                    
                    if boton_pulsado:
                        env.step(boton_pulsado.accion)
                    elif env.fase_actual == "DESCARTE":
                        # Manejar clic en cartas
                        x, y = agent_positions[0]
                        for j in range(4):
                            carta_rect = pygame.Rect(x - 120 + j * 70, y, 60, 100)
                            if carta_rect.collidepoint(mouse_pos):
                                env.step(10 + j)
                                break
        
        # Lógica de IA
        current_agent = env.agent_selection
        if current_agent != jugador_humano:
            pygame.time.delay(300)  # Pequeña pausa para ver el turno de la IA
            
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
                
                action = 0
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