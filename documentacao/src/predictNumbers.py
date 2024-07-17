import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

# Carregar o arquivo com a rede neural
arquivo = open('modelTraining\classificadorNumbers.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

# Converter para o modelo e carregar os seus pesos
classificador = model_from_json(estrutura_rede)
classificador.load_weights('modelTraining\classificadorNumbers.weights.h5')

# Função para prever a classe de uma imagem
def preverImagem(imagem):
    imagem = imagem.reshape(1, 28, 28, 1).astype('float32') / 255
    predicao = classificador.predict(imagem)
    return np.argmax(predicao, axis=1)[0]

# Interface gráfica para desenhar um número
class App(tk.Tk):
    # Inicializar a interface
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Desenhe algum número")
        self.canvas = tk.Canvas(self, width=200, height=200, bg="white")
        self.canvas.pack()
        
        self.button_predict = tk.Button(self, text="Prever", command=self.predict)
        self.button_predict.pack()
        
        self.button_clear = tk.Button(self, text="Limpar", command=self.clear)
        self.button_clear.pack()
        
        self.canvas.bind("<B1-Motion>", self.draw)
        
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

    # Função responsável por desenhar
    def draw(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    # Função responsável por chamar a rede neural e devolver qual número é
    def predict(self):
        self.image = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        self.image = ImageOps.invert(self.image)
        imagem_np = np.array(self.image)
        # plt.imshow(imagem_np, cmap='gray')    # Mostrar a imagem no formato do modelo
        # plt.show()
        classe_prevista = preverImagem(imagem_np)
        print(f'O número previsto é: {classe_prevista}')

    # Função responsável por limpar o quadro
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

app = App()
app.mainloop()
