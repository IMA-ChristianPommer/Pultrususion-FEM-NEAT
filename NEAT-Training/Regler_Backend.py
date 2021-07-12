import numpy as np
numofslices = 50
#import tensorflow as tf
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import interp1d
curinggates = [0.7, 0.6, 0.75, 0.75, 0.59, 0.62, 0.78,
               0.75, 0.66, 0.59, 0.80, 0.64, 0.66, 0.74,
               0.72, 0.6, 0.69, 0.8, 0.69, 0.76, 0.78, 0.79,
               0.73, 0.7, 0.74, 0.55]
gatetimes = 200


T_0 = 25*np.ones([50,50])
A_0 = 25*np.zeros([50,50])
t_slice = np.ones(numofslices)*10
p_slice = np.linspace(0, 1, numofslices)


T_Slices = [T_0]*numofslices
A_Slices = [A_0]*numofslices
alevels = np.arange(0, 1, 0.1)
tlevels = np.arange(25, 200, 25)




A=[]
T=[]
t=[]

#Reglernetzwerk generieren
#PreTrainedModel =tf.keras.models.load_model("Best_model0")


def Drawmodels(Drawqueue,dirstring):
    print("DRAW QUEUE Starting!")
    Roundfig, (ax_multipl1, ax_multipl2) = plt.subplots(2, 1, figsize=(15, 15))
    ax_multipl1.set_xlim([0,3500])
    ax_multipl1.set_ylim([0.5, 0.95])

    ax_multipl2.set_xlim([0,3500])
    ax_multipl2.set_ylim([0.05, 0.2])
    while(1):
        Header, Data1, Data2 = Drawqueue.get()
        if Header == 1: #Plot Data; Data1=x; Data2=y
            ax_multipl1.plot(Data1, color='0.1')
            ax_multipl2.plot(Data2,  color='0.1')
        elif Header == 2: #Data2=y; Data1=Data1
            ax_multipl1.hlines(y=Data1, xmin=Data2 * gatetimes + 50, xmax=(Data2 + 1) * gatetimes + 49)
        elif Header == 3:
            ax_multipl1.plot(Data1, color='orange')
            ax_multipl2.plot(Data2, color='0.1')
        elif Header == 4:
            Roundfig.savefig(dirstring + "Evalresults/" + str(Data1) + ".png", dpi=200)
            import tikzplotlib

            tikzplotlib.get_tikz_code(figure=Roundfig,filepath=dirstring + "Evalresults/" + str(Data1) + ".tex")

            ax_multipl1.cla()
            ax_multipl2.cla()
            ax_multipl1.set_xlim([0, 3500])
            ax_multipl1.set_ylim([0.5, 0.95])

            ax_multipl2.set_xlim([0, 3500])
            ax_multipl2.set_ylim([0.05, 0.2])

def Mutatemodel(initlayerweights, noisemaximum):
    np.random.seed();
    layerweights = copy.deepcopy(initlayerweights)
    #check mutation type
    Type = np.random.rand()
    if Type>=0.5:
        for i in range(len(layerweights)):
            layerweights[i] = layerweights[i] + (noisemaximum*(2*(np.random.rand(*layerweights[i].shape)-.5))**7)
    else:
        for i in range(len(layerweights)):
            layerweights[i] = layerweights[i] * (1+ (noisemaximum*(2*(np.random.rand(*layerweights[i].shape)-.5))**7))
    return(layerweights)

def calculatebadness(x, xt):
    return (np.abs(x-xt)**4)
    #Pre Training


def get_weights(model):
    return model.get('weights')


def get_goodness(model):
    return model.get('goodness')


def get_points(model):
    return model.get('points')

def GetTorA(A_Slices, T_Slices, p_slices, xpos, zpos):
    A_IN = []
    T_IN = []
    p_IN = []
    A_IN.append(0)
    T_IN.append(25)
    p_IN.append(0)
    for i in range(len(A_Slices)):
        A_IN.append(A_Slices[i][xpos, 0])
        T_IN.append(T_Slices[i][xpos, 0])
        p_IN.append(p_slices[i])


    A_NEW = interp1d(p_IN, A_IN, fill_value=0)
    T_NEW = interp1d(p_IN, T_IN, fill_value=0)

    A_OUT = A_NEW(zpos).mean()
    T_OUT = T_NEW(zpos).mean()

    return (A_NEW(zpos), T_NEW(zpos))

def InterpolatedTimeStep(Pult_Model, T_Slices, A_Slices, t_slice, p_slice,Geschw, dt_target):

    A_Slices10s, T_Slices10s, t_slice10s, p_slice10s = Makestep_Modelslice(Pult_Model, T_Slices, A_Slices, t_slice, p_slice, Geschw, False)

    for n in range(len(A_Slices10s)):

        A_new = (-A_Slices[n] + A_Slices10s[n]) * dt_target / 10.0 + A_Slices[n]
        T_new = (-T_Slices[n] + T_Slices10s[n]) * dt_target / 10.0 + T_Slices[n]
        p_new = (-p_slice[n]  +  p_slice10s[n]) * dt_target / 10.0 +  p_slice[n]
        t_new = t_slice[n] + dt_target

        if (len(t_slice) - np.count_nonzero(t_slice) > 1):
            A = 1
        if p_new > 1:
            A_Slices[n] = np.copy(A_0)
            T_Slices[n] = np.copy(T_0)
            p_slice[n] = 0
            t_slice[n] = 0
        elif ((t_slice[n] == 0) and (np.logical_and(t_slice > 0,t_slice <= 10)).any()):
            A_Slices[n] = np.copy(A_0)
            T_Slices[n] = np.copy(T_0)
            p_slice[n] = 0
            t_slice[n] = 0
        else:
            A_Slices[n] = A_new
            T_Slices[n] = T_new
            p_slice[n] = p_new
            t_slice[n] = t_new



    return (A_Slices, T_Slices, t_slice, p_slice)
    #Start numbere

def Makestep_Modelslice(Pult_Model,T_Slices,A_Slices,t_slice,p_slice,Geschw, RESET_ON_END=True):
    Datain = []
    T_IN = []
    A_IN = []

    for i in range(numofslices):
        Datain.append(np.array([Geschw, t_slice[i]*1.0/500]))
        T_IN.append((T_Slices[i].flatten())/255)
        A_IN.append((A_Slices[i].flatten()))

    D_IN = np.array(Datain, dtype=np.float32)
    A_IN = np.array(A_IN,   dtype=np.float32)
    T_IN = np.array(T_IN,   dtype=np.float32)


    (A_OUT, T_OUT) = Pult_Model((A_IN, T_IN, D_IN))

    A_OUT = A_OUT.numpy()
    T_OUT = T_OUT.numpy()

    A_OUT_RESULT = [1]*numofslices
    T_OUT_RESULT = [1]*numofslices

    for i in range(numofslices):
        A_OUT_RESULT[i] = (np.reshape(A_OUT[i], [50, 50]))
        T_OUT_RESULT[i] = (np.reshape(T_OUT[i], [50, 50]) * 255)

    if RESET_ON_END:
        for i in range(numofslices):
            if (p_slice[i] > 1):
                A_OUT_RESULT[i] = np.copy(A_0)
                T_OUT_RESULT[i] = np.copy(T_0)
                p_slice[i] = 0
                t_slice[i] = 0
                print(str(i) + " Bigger 1")
            elif(p_slice[i] > 0):
                t_slice[i] = t_slice[i] + 10.0
                p_slice[i] = p_slice[i] + Geschw * 10.0
            else:
                if (np.logical_and(p_slice > 0, p_slice < minslicedistance).any()):
                    A_OUT_RESULT[i] = np.copy(A_0)
                    T_OUT_RESULT[i] = np.copy(T_0)
                    p_slice[i] = 0
                    t_slice[i] = 0
                else:
                    t_slice[i] = t_slice[i] + 10.0
                    p_slice[i] = p_slice[i] + Geschw * 10.0
                    print(str(i) + " Starting")
    else:
        t_slice = t_slice + 10.0
        p_slice = p_slice + Geschw * 10.0

    return(A_OUT_RESULT, T_OUT_RESULT, t_slice, p_slice)

def Trainingloop(Model, net, drawqueue, draw, speed):
    # graph.finalize()
    ModelAIn = []
    tIn = []
    Points = 0
    # Localmodel = tf.keras.models.clone_model(PreTrainedModel)
    # Localmodel.set_weights(PreTrainedModel.get_weights())
    A_Slices = copy.deepcopy(A_Slices_Start)
    T_Slices = copy.deepcopy(T_Slices_Start)
    t_Slices = copy.deepcopy(t_slice_Start)
    p_Slices = copy.deepcopy(p_slice_Start)
    # Pre_Generation
    DrawA = []
    DrawS = []

    for i in range(50):
        (A_Slices, T_Slices, t_Slices, p_Slices) = InterpolatedTimeStep(Model, T_Slices, A_Slices, t_Slices, p_Slices,
                                                                        0.1 / 60, 10*speed)
        A_new, T_new = GetTorA(A_Slices, T_Slices, p_Slices, 25, 0.8)
        ModelAIn.append(A_new.mean())
        DrawA.append(A_new.mean())
        DrawS.append(0.1)
    # Testrung
    ModelAIn = ModelAIn[-19:]
    n = 0
    for Curing in curinggates:
        drawqueue.put([2,Curing,n ])
        for i in range(gatetimes):
            v_model = net.activate((ModelAIn + [Curing]))
            v_model = v_model[0]
            v_new = np.clip(v_model, 0.05, 0.2)
            v_model = np.clip(v_model, 0.05, 0.2)
            ModelAIn = ModelAIn[1:]

            (A_Slices, T_Slices, t_Slices, p_Slices) = InterpolatedTimeStep(Model, T_Slices, A_Slices, t_Slices,
                                                                            p_Slices, v_new / 60, 10)
            A_new, T_new = GetTorA(A_Slices, T_Slices, p_Slices, 25, 0.8)
            ModelAIn.append(A_new.mean())
            DrawA.append(A_new.mean())
            DrawS.append(v_new)

            if (0.05 <= v_model <= 0.2):
                Points += 1 + np.abs(Curing - A_new.mean())
            else:
                if (v_model < 0.05):
                    Points += 1 / (1 + np.abs(v_model - 0.05))
                elif (v_model > 0.2):
                    Points += 1 / (1 + np.abs(v_model - 0.2))
                break


        if (0.05 <= v_model <= 0.2):
            if np.abs(A_new.mean() - Curing) < 0.01:
                Points += 1000
            else:
                dist = np.abs(A_new.mean() - Curing)
                Points += 1000 - dist * 1000
                break
        else:
            break
        n += 1

    if draw:
        drawqueue.put([1,DrawA, DrawS])
    # if (v_model >= 0.05) and (v_model <= 0.2):
    #      if (v_model < 0.05): Points += np.abs(v_model-0.05)*100+100000
    #     elif (v_model > 0.2): Points += np.abs(v_model-0.2)*100+100000



    return (Points)

Data = np.load("Startingslices.npz")

A_Slices_Start = Data["A_Slices_Start"]
T_Slices_Start = Data["T_Slices_Start"]
t_slice_Start = Data["t_slice_Start"]
p_slice_Start = Data["p_slice_Start"]
