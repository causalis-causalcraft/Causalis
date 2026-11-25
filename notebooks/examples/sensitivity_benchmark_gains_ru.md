$$
\begin{aligned}
\mathrm{passed}
&=\mathrm{preconditions\_passed}
\land(\mathcal B\ne\varnothing)
\land\bigwedge_{b\in\mathcal B}\mathrm{passed}_{b,\mathrm{primary}},\\
\mathrm{status}
&=\begin{cases}
\mathrm{PASS},&\mathrm{passed},\\
\mathrm{FAIL},&\neg\mathrm{passed}.
\end{cases}
\end{aligned}
$$
Соблюден Overlap и Unconfoundedness, SUTVA. Существует B. Статус passed у всех кластеров в провверке


$$
\mathrm{passed}_{b,z}
=\mathrm{scenario\_valid}_{b,z}\land
\begin{cases}
C_z^->\tau,&d=\mathrm{positive},\\
C_z^+<\tau,&d=\mathrm{negative}.
\end{cases}
$$
Статус passed: Bias-aware доверительный интервал больше decision_threshold
$$
\mathrm{scenario\_valid}_{b,z}
=V_L\land\operatorname{finite}(B_0F_z,B_0s_z,C_z^-,C_z^+)\land
\begin{cases}
V_{\mathrm{calibration}},&z\in\{\mathrm{primary},\mathrm{stress}\},\\
V_{\mathrm{strengths}},&z=\mathrm{adversarial}.
\end{cases}
$$
\(V_L\) - есть оценка ATTE, посчитаны кэффы и есть параметры на входе.
Учитываем sampling uncertainty. Обе границы рассчитаны по выборке. Поэтому нижнюю опускаем ещё на \(q\,\mathrm{SE}_{-}\), а верхнюю поднимаем на \(q\,\mathrm{SE}_{+}\). У каждой границы свой standard error. Для 95% CI в нашем примере \(q\approx1{,}96\).
$$
\mathrm{bias\_aware\_ci}_z
=\left[C_z^-,C_z^+\right]
=\left[\hat\theta_L-B_0s_z-q\,\mathrm{SE}_{-}(s_z),\quad
\hat\theta_L+B_0s_z+q\,\mathrm{SE}_{+}(s_z)\right].
$$
\(B_0s_z\) — насколько эффект может сдвинуться при выбранном scenario.
$$
\begin{aligned}
\tau&=\mathrm{decision\_threshold},\qquad
q=\Phi^{-1}(1-\alpha/2),\qquad \alpha_{\mathrm{default}}=0{,}05,\\
d&=\begin{cases}
\mathrm{direction},&\mathrm{direction}\in\{\mathrm{positive},\mathrm{negative}\},\\
\mathrm{positive},&\mathrm{direction}=\mathrm{auto}\land\hat\theta_L\ge\tau,\\
\mathrm{negative},&\mathrm{direction}=\mathrm{auto}\land\hat\theta_L<\tau.
\end{cases}
\end{aligned}
$$
Проверка на знак

$$
B_0=\sqrt{\sigma_L^2\nu_L^2},\qquad
F_z=\sqrt{c_{y,z}\frac{c_{d,z}}{1-c_{d,z}}},\qquad
s_z=|\rho_z|F_z.
$$
\(B_0\) — базовый масштаб bias для long model. \(\sigma_L^2\) показывает, сколько ошибки остаётся после прогноза outcome. \(\nu_L^2\) характеризует масштаб весов, которыми ATTE учитывает эти ошибки. Вместе они задают масштаб в единицах outcome.
\(F_z\) — сила hidden confounding в scenario \(z\). \(c_{y,z}\) задаёт силу со стороны outcome, а \(c_{d,z}\) — со стороны treatment через Riesz moment. Для treatment используется отношение \(c_{d,z}/(1-c_{d,z})\). Если сила с любой стороны равна нулю, \(F_z=0\).
\(s_z\) — сила с учётом \(\rho_z\). \(|\rho_z|\) показывает, какая доля потенциального bias реализуется при заданных силах. При \(|\rho_z|=1\) берём весь потенциал \(F_z\), при \(|\rho_z|=0\) получаем нулевой сдвиг.
$$
\begin{aligned}
u_{\pm,i}(s)&=\psi_{\theta,i}\pm s\,\psi_{B_0,i},\\
\mathrm{SE}_{\pm}(s)
&=\sqrt{\frac{\sum_{i=1}^{n}\bigl(u_{\pm,i}(s)-\overline{u_{\pm}(s)}\bigr)^2}{n(n-1)}},\\
\mathrm{SE}_L&=\mathrm{SE}_{-}(0)=\mathrm{SE}_{+}(0).
\end{aligned}
$$

\(u_{-,i}\) и \(u_{+,i}\) — вклад строки в неопределённость нижней и верхней границ. Берём вклад строки в оценку ATTE и вычитаем или прибавляем вклад в bias bound с множителем \(s\). Для каждой границы считаем standard deviation этих вкладов и делим на \(\sqrt n\). При \(s=0\) обе формулы дают обычный standard error для ATTE.

$$
\psi_{B_0,i}=\tfrac12\left[
\sqrt{\frac{\nu_L^2}{\sigma_L^2}}\,(e_{L,i}^2-\sigma_L^2)
+\sqrt{\frac{\sigma_L^2}{\nu_L^2}}\,(v_{L,i}-\nu_L^2)
\right],
\qquad
\psi_{\theta,i}=b_{L,i}-\frac{D_i}{\bar D}\hat\theta_L.
$$

\(\psi_{\theta,i}\) показывает вклад строки в неопределённость оценки ATTE. \(\psi_{B_0,i}\) показывает её вклад в неопределённость базового масштаба bias. Для этого смотрим, насколько квадрат ошибки outcome и вклад в Riesz moment у этой строки отличаются от своих средних.

$$
\begin{aligned}
V_L={}&(n\ge2)\land\operatorname{finite}(\hat\theta_L,\sigma_L^2,\nu_L^2,\mathrm{SE}_L)\\
&\land(\sigma_L^2>0)\land(\nu_L^2>0)\land(\mathrm{SE}_L\ge0)\\
&\land(\psi_{\theta},\ e_L^2-\sigma_L^2,\ v_L-\nu_L^2\in\mathbb R^n).
\end{aligned}
$$

\(V_L\) проверяет, что данных из long model хватает для расчёта. Нужны хотя бы две строки, конечные ATTE и standard error, положительные \(\sigma_L^2\) и \(\nu_L^2\) для калибровки и конечный вклад каждой строки.

$$
(c_{y,z},c_{d,z},\rho_z)=
\begin{cases}
(cf_y,cf_d,\rho),&z=\mathrm{primary},\\
\left(\min(k\,cf_y,1-10^{-12}),\dfrac{k\,cf_d}{1-cf_d+k\,cf_d},\rho\right),&z=\mathrm{stress},\\
(cf_y,cf_d,1),&z=\mathrm{adversarial},
\end{cases}
\qquad
k=\mathrm{stress\_multiplier},\quad k_{\mathrm{default}}=2.
$$

В `primary` берём силы и \(\rho\) из benchmark refit. В `stress` при \(k=2\) удваиваем \(cf_y\) и treatment odds \(cf_d/(1-cf_d)\). \(cf_y\) при этом ограничен чуть ниже единицы, а odds переводятся обратно в \(cf_d\). В `adversarial` оставляем исходные силы и ставим \(\rho=1\).

$$
\begin{aligned}
V_{\mathrm{calibration}}&=V_{\mathrm{strengths}}\land\operatorname{finite}(\rho),\\
V_{\mathrm{strengths}}&=
\mathrm{short\_refit\_ok}
\land\operatorname{finite}(\hat\theta_L,\hat\theta_S,cf_{y,\mathrm{raw}},cf_{d,\mathrm{raw}})\\
&\quad\land(\sigma_L^2,\sigma_S^2,\nu_L^2,\nu_S^2\in(0,\infty))
\land(cf_y,cf_d\in[0,1)).
\end{aligned}
$$

\(V_{\mathrm{strengths}}\) проверяет, что short model успешно переобучена, исходные значения конечны, а обе силы лежат в диапазоне \([0,1)\). \(V_{\mathrm{calibration}}\) дополнительно требует конечное \(\rho\).

$$
cf_y=\operatorname{clip}_{[0,1]}(cf_{y,\mathrm{raw}}),\qquad
cf_d=\operatorname{clip}_{[0,1]}(cf_{d,\mathrm{raw}}).
$$

Если raw strength ниже нуля, ставим ноль. Если она выше единицы, ставим единицу. Единица помечает benchmark как invalid.

$$
\rho=
\begin{cases}
\operatorname{clip}_{[-1,1]}(\rho_{\mathrm{raw}}),
&G_y>0\land G_d>0\land\operatorname{finite}(\rho_{\mathrm{raw}}),\\
\operatorname{sign}(\hat\theta_S-\hat\theta_L),&G_y\le0\lor G_d\le0,\\
\mathrm{NaN},&\text{otherwise}.
\end{cases}
$$

Когда оба gains положительны, ограничиваем рассчитанное \(\rho_{\mathrm{raw}}\) диапазоном от \(-1\) до \(1\). Если хотя бы один gain нулевой или отрицательный, берём знак изменения ATTE. Тогда одна из сил равна нулю, и bias bound имеет нулевую ширину.

$$
cf_{y,\mathrm{raw}}=\frac{G_y}{\sigma_L^2},\qquad
cf_{d,\mathrm{raw}}=\frac{G_d}{\nu_S^2},\qquad
\rho_{\mathrm{raw}}=\frac{\hat\theta_S-\hat\theta_L}{\sqrt{G_yG_d}}
\quad(G_y>0,\ G_d>0).
$$

Для outcome делим прирост ошибки на ошибку long model. Для treatment делим изменение Riesz moment на его значение в short model. Для \(\rho_{\mathrm{raw}}\) сравниваем сдвиг ATTE с масштабом \(\sqrt{G_yG_d}\). Например, без `support_tickets_90d` ATTE снизился с \(13{,}43\) до \(5{,}47\). Получаем \(\rho_{\mathrm{raw}}\approx -7{,}96/45{,}58\approx -0{,}175\).

$$
G_y=\sigma_S^2-\sigma_L^2,\qquad
G_d=\nu_L^2-\nu_S^2.
$$

\(G_y\) показывает, насколько выросла ошибка прогноза outcome после исключения группы. \(G_d\) показывает, насколько Riesz moment в long model выше, чем в short model. Поэтому порядок вычитания для них разный.

$$
\hat\theta_M=\overline{b_M},\qquad
\sigma_M^2=\overline{e_M^2},\qquad
\nu_M^2=\overline{v_M},\qquad
M\in\{L,S\}.
$$

В каждой из двух моделей усредняем три величины по строкам. Из \(b_M\) получаем оценку ATTE, из \(e_M^2\) — средний квадрат ошибки прогноза outcome, из \(v_M\) — Riesz moment.

$$
\begin{aligned}
b_{M,i}&=r_{M,i}\bigl(Y_i-\hat g_{M,0,i}\bigr),\\
e_{M,i}&=Y_i-D_i\hat g_{M,1,i}-(1-D_i)\hat g_{M,0,i},\\
v_{M,i}&=\frac{2D_i}{\bar D^2(1-\hat m_{M,i})}-r_{M,i}^2,\\
r_{M,i}&=\frac{D_i}{\bar D}-\frac{(1-D_i)\hat m_{M,i}}{\bar D(1-\hat m_{M,i})}.
\end{aligned}
$$

\(b_{M,i}\) — вклад строки в ATTE. Берём разницу между наблюдаемым outcome и прогнозом без treatment, затем умножаем на Riesz weight \(r_{M,i}\). У treated этот вес равен \(1/\bar D\). У control он отрицательный, а его величина растёт вместе с propensity score.

\(e_{M,i}\) — ошибка прогноза для фактической группы строки. \(v_{M,i}\) — вклад строки в оценку квадратичного момента Riesz weights.

$$
\begin{aligned}
\hat m_{M,i}&=\operatorname{clip}_{[0{,}01;\,0{,}99]}
\!\left(\widehat P_M(D_i=1\mid X_{M,i})\right),\\
\hat g_{M,j,i}&=\widehat{\mathbb E}_M[Y_i\mid D_i=j,X_{M,i}],\qquad j\in\{0,1\},\\
\operatorname{clip}_{[a,b]}(x)&=\min\{b,\max(a,x)\},\qquad
\overline{f}=\frac1n\sum_{i=1}^{n}f_i,\qquad
\bar D=\frac1n\sum_{i=1}^{n}D_i.
\end{aligned}
$$

\(\hat m_{M,i}\) — прогноз вероятности treatment. Значения ниже \(0{,}01\) поднимаем до \(0{,}01\), значения выше \(0{,}99\) опускаем до \(0{,}99\). Все строки остаются в расчёте. \(\hat g_{M,0,i}\) и \(\hat g_{M,1,i}\) — прогнозы outcome для control и treated, а \(\bar D\) — доля treated в выборке.

$$
\mathcal X_S=\mathcal X_L\setminus b,\qquad b\in\mathcal B,\qquad
D_i\in\{0,1\},\qquad i\in\{1,\ldots,n\},\qquad
\mathrm{score}=\mathrm{ATTE}.
$$

Для одной benchmark group \(b\) переобучаем short model без признаков этой группы. Используем те же строки и тот же ATTE score, что и в long model.

## Интуиция и обозначения

Исключаем наблюдаемую benchmark group и используем изменения после refit как ориентир для силы hidden confounding. При пройденных внешних проверках `PASS` означает, что каждый `primary` bias-aware CI целиком по нужную сторону порога.

- \(L\), \(S\) — модели со всеми признаками и без группы \(b\). \(z\) — scenario. \(\hat\theta_L\) — исходная оценка ATTE.
- \(Y\) — outcome, \(D\) — treatment, \(X\) — признаки. \(\hat g\) — прогноз outcome, \(\hat m\) — propensity score.
- \(\sigma^2\) — средний квадрат ошибки прогноза outcome. \(\nu^2\) — квадратичный момент Riesz weights.
- \(c_{y,z}\), \(c_{d,z}\) — силы по outcome и treatment. \(|\rho_z|\) — доля потенциального bias.
- \(B_0s_z\) — сдвиг из-за hidden confounding. \(q\,\mathrm{SE}\) — добавка за sampling uncertainty. \([C_z^-,C_z^+]\) — итоговый CI, \(\tau\) — `decision_threshold`.
