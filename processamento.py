#Processamento Imagem Colorida
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plot
import math

img = cv.imread('carro.jpg')

plot.figure(1) 
plot.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plot.title('Imagem Original',fontweight ="bold")

def processamentoColorido(img,funcao):
    (imgB,imgG,imgR) = cv.split(img)
    imgSaidaB = funcao(imgB)
    imgSaidaG = funcao(imgG)
    imgSaidaR = funcao(imgR)
    imgSaida = cv.merge((imgSaidaR,imgSaidaG,imgSaidaB))
    return imgSaida

def criarHistograma(img):
    quantLinhas = np.size(img, 0)
    quantColunas = np.size(img, 1)
    
    #calculando o histograma
    hist = np.zeros(256)
    for linha in range(quantLinhas):
        for coluna in range(quantColunas):
            valor = img[linha, coluna]
            hist[ valor ] += 1
    
    #calculando o histograma relativo
    histRel = np.zeros(256)
    for i in range(256):
        histRel[i] = hist[i] / (quantLinhas * quantColunas)

    #calculando o histograma relativo acumulado
    histRelAcc = np.zeros(256)
    histRelAcc[0] = histRel[0]
    for i in range(1, 256):
        histRelAcc[i] = histRelAcc[i-1] + histRel[i]

    
    return (hist, histRel, histRelAcc)
    
def negativo(img):
    quantLinhas = np.size(img,0)
    quantColunas = np.size(img,1)
    imgSaida = np.zeros((quantLinhas,quantColunas),dtype='uint8')
    for l in range(quantLinhas):
        for c in range(quantColunas):
            imgSaida[l,c] = 255 - img[l,c]
    return imgSaida
(imgB,imgG,imgR) = cv.split(img)
imgSaidaB = negativo(imgB)
imgSaidaG = negativo(imgG)
imgSaidaR = negativo(imgR)
imgSaida = cv.merge((imgSaidaR,imgSaidaG,imgSaidaB))
plot.figure(2) 
plot.imshow(imgSaida)
plot.title('Imagem Negativa',fontweight ="bold")

def equalizarImagem(img):
    quantLinhas = np.size(img, 0)
    quantColunas = np.size(img, 1)
    (hist, histRel, histRelAcc) = criarHistograma(img)
    Eq = np.zeros(256)
    for i in range(256):
        Eq[i] = 255 * histRelAcc[i]

    imgSaida = np.zeros( (quantLinhas, quantColunas), dtype='uint8' )
    for linha in range(quantLinhas):
        for coluna in range(quantColunas):
            valor = img[linha, coluna]
            imgSaida[linha, coluna] = Eq[ valor ]

    return imgSaida
imgSaida = processamentoColorido(img,equalizarImagem)
plot.figure(3) 
plot.imshow(imgSaida)
plot.title('Imagem Equalizada',fontweight ="bold")

def relaceLog(img):
    quantLinhas = np.size(img, 0)
    quantColunas = np.size(img, 1)

    histLog = np.zeros(256)
    for i in range(256):
        valor = 105.866 * math.log10(i + 1)
        if (valor < 0):
            histLog[i] = 0
        elif(valor > 255):
            histLog[i] = 255
        else:
            histLog[i] = valor

    imgSaida = np.zeros( (quantLinhas, quantColunas), dtype='uint8' )
    for linha in range(quantLinhas):
        for coluna in range(quantColunas):
            valor = img[linha, coluna]
            imgSaida[linha, coluna] = histLog[ valor ]

    return imgSaida
imgSaida = processamentoColorido(img,relaceLog)
plot.figure(4) 
plot.imshow(imgSaida)
plot.title('Relace Log',fontweight ="bold")

def relacePotencia(img, gamma=2):
    quantLinhas = np.size(img, 0)
    quantColunas = np.size(img, 1)

    histPotencia = np.zeros(256)
    for i in range(256):
        valor = math.pow(255, 1-gamma) * math.pow(i, gamma)
        if (valor < 0):
            histPotencia[i] = 0
        elif(valor > 255):
            histPotencia[i] = 255
        else:
            histPotencia[i] = valor


    imgSaida = np.zeros( (quantLinhas, quantColunas), dtype='uint8' )
    for linha in range(quantLinhas):
        for coluna in range(quantColunas):
            valor = img[linha, coluna]
            imgSaida[linha, coluna] = histPotencia[ valor ]

    return imgSaida
imgSaida = processamentoColorido(img,relacePotencia)
plot.figure(5) 
plot.imshow(imgSaida)
plot.title('Relace Potencia',fontweight ="bold")

def relaceExponencial(img):
    quantLinhas = np.size(img, 0)
    quantColunas = np.size(img, 1)
    histExp = np.zeros(256)
    for i in range(256):
        valor = math.exp(i / 45.986) - 1
        if (valor < 0):
            histExp[i] = 0
        elif(valor > 255):
            histExp[i] = 255
        else:
            histExp[i] = valor

    imgSaida = np.zeros( (quantLinhas, quantColunas), dtype='uint8' )
    for linha in range(quantLinhas):
        for coluna in range(quantColunas):
            valor = img[linha, coluna]
            imgSaida[linha, coluna] = histExp[ valor ]

    return imgSaida
imgSaida = processamentoColorido(img,relaceExponencial)
plot.figure(6) 
plot.imshow(imgSaida)
plot.title('Relace Exponencial',fontweight ="bold")

plot.show()