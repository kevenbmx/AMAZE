# O programa que ajuda quem tem labirintite

# backlog

# metodo melhor pra encontrar grossura - tirar o mini bfs e colocar algo mais rapido
# metodo pra identificar quando redimensionar a imagem
# fechar o labirinto
# testar
# artigo

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import collections
import imageio

BRANCO = 255
PRETO = 0

def redimensionar(image, inicio, fim):
    if(image.shape[0] < 300):
        return(image, inicio, fim)
    
    pct = 1/(image.shape[1] / 300)
    
    img=image.copy()
    width = int(img.shape[1] * pct)
    height = int(img.shape[0] * pct)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    inicio = [int(i * pct) for i in inicio]
    fim = [int(i * pct) for i in fim]
    
    return(img,inicio,fim)
    

def encontraGrossura(maze_image,inicio):
    img = maze_image.copy()
    fila = []
    fila.append([inicio])
    while fila:
        
        # if((len(fila) > 90000000) & (len(fila) % 2 == 0) ):
            # print(len(fila))
            # plt.imshow(img)
            # plt.show()
        
        caminho = fila.pop(0)
        atual = caminho[-1]
        
        for vizinho in vizinhos(atual[0],atual[1]):
            novo_caminho = list(caminho)
            novo_caminho.append(vizinho)
            if(img[vizinho[0],vizinho[1]] < 50):
                return(len(caminho))
                break
            img[vizinho[0],vizinho[1]] = 50
            fila.append(novo_caminho)

                                
def amaze(maze_image, inicio, fim):
    # redimensionar imagem
    maze_image,inicio,fim = redimensionar(maze_image, inicio, fim)
    
    print(inicio, ',',fim)
    original = maze_image.copy()
    
    maze_image = cv2.Canny(maze_image, 50, 150)
    maze_image =  cv2.bitwise_not(maze_image)
    # tratamento da imagem
    #maze_image = cv2.cvtColor(maze_image,cv2.COLOR_BGR2GRAY)
    
    #maze_image = cv2.adaptiveThreshold(maze_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #retval2,maze_image = cv2.threshold(maze_image,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
    
    plt.imshow(maze_image)
    plt.show()
    

    
    # engrossamento das linhas
    solucao = None
    
    grossuras = [encontraGrossura(maze_image,inicio)]
    grossuras.append(encontraGrossura(maze_image,fim))
    grossura = np.min(grossuras)
    print(grossuras)
    
    while solucao is None and grossura >= 0:
        
        maze_image_eroded = cv2.erode(maze_image, cv2.getStructuringElement(cv2.MORPH_RECT,(grossura,grossura)), iterations=1)
        
        plt.imshow(maze_image_eroded)
        plt.show() 
        
        # solucao
        solucao = (bfs(maze_image_eroded, inicio, fim))    
        
        if(solucao is not None):
            for pixel in solucao:
                original[pixel[0], pixel[1]] = [255,0,0]
                for vizinho in vizinhosN(pixel[0],pixel[1],1):
                    if(maze_image[vizinho[0], vizinho[1]] > 125):
                        original[vizinho[0], vizinho[1]] = [255,0,0]

            plt.imshow(original)
            plt.show()
        
        grossura = int(grossura / 2)  
        if(grossura < 1):
            grossura = 0
            

def bfs(maze_image,inicio,fim):
    fila = []
    fila.append([inicio])
    gif = []
        
    while fila:
        caminho = fila.pop(0)
        #print('print',caminho[-1])
        atual = caminho[-1]

        if(atual == fim):
            #fila.clear()
            return caminho
        
        
        for vizinho in vizinhos(atual[0],atual[1]):
            if(maze_image[vizinho[0],vizinho[1]] == BRANCO):
                novo_caminho = list(caminho)
                novo_caminho.append(vizinho)
                maze_image[vizinho[0],vizinho[1]] = 50
                fila.append(novo_caminho)


def vizinhos(x,y):
    return[[x, y-1],
           [x - 1, y], [x + 1, y],
           [x, y+1]]

def vizinhosDiag(x,y):
    return[[x-1, y-1], [x, y-1], [x+1, y-1],
           [x - 1, y], [x + 1, y],
           [x - 1, y+1], [x, y+1], [x + 1, y+1]]

def vizinhosN(x,y,n):
    vizinhos = []
    for vx in range(x-n, x+n):
        for vy in range(y-n, y+n):
            vizinhos.append([vx,vy])
    return(vizinhos)

amaze(cv2.imread('maze1.png'), [13,156], [320,172])

#amaze(cv2.imread('maze2.png'), [203, 130], [293, 285])

#amaze(cv2.imread('maze3.png'), [27,269], [345,267])

#amaze(cv2.imread('maze4.png'),  [10,174], [203,155])

#amaze(cv2.imread('maze5.png'), [153,495],[92,471])

#amaze(cv2.imread('mazePapel.png'), [188,159], [800,392])

#amaze(cv2.imread('macaco.jpg'), [247,197], [631,424])