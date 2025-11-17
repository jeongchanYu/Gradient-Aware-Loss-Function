import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


x_max = 10
x_true = 1.0
# x_max = 100
# x_true = 10.0
x_pred = np.linspace(0.0, x_max, 1000)
glsd1 = - x_pred * (np.log(x_true) - np.log(x_pred) + 1) + x_true
glsd2 = - x_pred * (np.square(np.log(x_true) - np.log(x_pred) + 1) + 1) + 2 * x_true
glsd2 = np.where(x_true >= x_pred, glsd2, -glsd2)

plt.figure(figsize=(4, 4), dpi=300)
plt.plot(x_pred, glsd1, linewidth=1.2, label='GLSD1')
plt.plot(x_pred, glsd2, linewidth=1.2, label='GLSD2')
plt.grid(True, which="both", linewidth=0.4)

plt.xlabel("s_pred")
plt.ylabel("loss")
plt.xlim(0, x_max)
# plt.ylim(-3, 3)
# plt.legend()
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(4, 4), dpi=300)
plt.plot(x_pred, glsd1, linewidth=1.2, label='GLSD1')
plt.plot(x_pred, glsd2, linewidth=1.2, label='GLSD2')
plt.grid(True, which="both", linewidth=0.4)

plt.xlabel("s_pred")
plt.ylabel("loss")
plt.xscale('log')
plt.xlim(x_max/100, x_max)
# plt.ylim(-5, 5)
# plt.legend()
plt.tight_layout()
plt.show()
plt.close()


glsd1_grad = np.gradient(glsd1, x_pred)  # df/dx
glsd2_grad = np.gradient(glsd2, x_pred)  # df/dx

plt.figure(figsize=(4, 4), dpi=300)
plt.plot(x_pred, glsd1_grad, linewidth=1.2, label='GLSD1')
plt.plot(x_pred, glsd2_grad, linewidth=1.2, label='GLSD2')
plt.grid(True, which="both", linewidth=0.4)

plt.xlabel("s_pred")
plt.ylabel("gradient")
plt.xlim(0, x_max)
plt.ylim(-5, 5)
# plt.legend()
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(4, 4), dpi=300)
plt.plot(x_pred, glsd1_grad, linewidth=1.2, label='GLSD1')
plt.plot(x_pred, glsd2_grad, linewidth=1.2, label='GLSD2')
plt.grid(True, which="both", linewidth=0.4)

plt.xlabel("s_pred")
plt.ylabel("gradient")
plt.xscale('log')
plt.xlim(x_max/100, x_max)
plt.ylim(-5, 5)
# plt.legend()
plt.tight_layout()
plt.show()
plt.close()