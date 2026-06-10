# Street pattern и network-distance accessibility до сервисов

## Исходная постановка

Вопрос:

> Как использовать знание о street pattern, чтобы улучшать попадание жителей / адресов / участков в заданный радиус доступности до сервисов, если доступность считается только по network distance?

Уточнение:

- Метрика доступности: **только network distance**.
- Euclidean radius / straight-line distance / “as the crow flies” не используется как метрика доступности.
- Выводы должны опираться на источники, без свободного синтеза сверх них.

---

## 1. Street pattern нужен как объяснение, почему network catchment меньше или больше

Human Transit показывает, что “радиус спроса” вокруг остановки или сервиса нельзя считать по прямой, потому что люди идут по сети улиц и путей. Фактическая зона доступности определяется **street/path network**, а не кругом “as the crow flies”. В примере с cul-de-sac дома могут быть близко по прямой, но далеко по фактическому пешему маршруту.

Source:  
https://humantransit.org/2010/05/culdesac-hell-and-the-radius-of-demand.html

Boeing формулирует это через **circuity**: отношение network distance к straight-line distance является важной характеристикой структуры уличной сети и транспортной эффективности. В его работе подчёркивается ценность network-based distances / times вместо straight-line measures при анализе urban travel and access.

Source:  
https://arxiv.org/abs/1708.00836

Следовательно, по источникам можно сказать:

> Street pattern используется для выявления мест, где заданный порог доступности по сети не достигается из-за конфигурации сети, а не из-за геометрической удалённости сервиса.

---

## 2. Для cul-de-sac основной источник интервенций — pedestrian/cycle connectivity

Global Designing Cities Initiative в разделе про pedestrian networks пишет, что нужно создавать pedestrian links, чтобы сокращать walking routes. Streets and paths that end in cul-de-sacs should be extended to connect to nearby streets. Также рекомендуется создавать pedestrian links through large blocks для более мелкозернистой ткани и лучшей connectivity.

Source:  
https://globaldesigningcities.org/publication/global-street-design-guide/designing-streets-people/designing-for-pedestrians/pedestrian-networks/

Sustainable City Code отдельно формулирует это для **culs-de-sac**: cul-de-sacs увеличивают расстояние, которое считается комфортным для ходьбы, а pedestrian connectivity помогает уменьшать время и расстояние до destinations.

Source:  
https://sustainablecitycode.org/brief/pedestrian-connectivity-through-culs-de-sac/

UC Davis study on non-motorized accessibility and connectivity показывает, что **off-street pedestrian links** могут существенно влиять на pedestrian connectivity and accessibility. В тексте указано, что suburban areas с pedestrian network based on pathways, parks, and greenbelts могут иметь более высокий уровень connectivity/accessibility, чем это показала бы только street network.

Source:  
https://itspubs.ucdavis.edu/download_pdf.php?id=1665

Следовательно, применительно к заданному network-distance порогу:

> В cul-de-sac / low-connectivity pattern надо искать места, где короткий pedestrian/cycle link переводит адреса из “за пределами порога” в “внутри порога”.

---

## 3. Для grid проблема реже в недостатке street pattern

В обзоре Zlatkovic et al. цитируется вывод Yi: **grid network provides better accessibility to destinations for pedestrians**, но при добавлении отдельных pedestrian trails cul-de-sac network тоже может повысить accessibility.

Source:  
https://wfrc.utah.gov/PublicInvolvement/InTheNews/AssessmentOfEffectsOfStreetConnectivity.pdf

WRI / road-safety guide описывает connected / grid networks как такие, которые дают пешеходам и велосипедистам более прямые маршруты, тогда как disconnected / cul-de-sac / superblock networks могут discouraging walking and bicycling.

Source:  
https://www.roadguidelines.ie/wp-content/uploads/2025/10/DMURS-C3-HiRes.pdf

Broward Complete Streets Guidelines описывают street network configuration как фактор, влияющий на базовые аспекты urban transportation; well-planned street networks рассматриваются как основа sustainable cities.

Source:  
https://www.browardmpo.org/images/WhatWeDo/completestreetsinitiative/broward_complete_streets_guidelines_parts/CH-4-Street-Networks-and-Classifications-final.pdf

По источникам можно аккуратно сказать:

> Если территория уже grid-like и адрес всё равно не попадает в заданный network-distance радиус до сервиса, то сам street pattern, вероятно, не главный ограничитель. Нужно проверять фактическое расположение сервиса, барьеры, переходы, качество pedestrian network или плотность сети сервисов.

Это не отдельный “сильный вывод” статьи про grid, а следствие из того, что grid обычно даёт более прямую сетевую доступность, чем disconnected patterns.

---

## 4. Warped parallel / mixed важны для PT-литературы

Статья Pasha, Rifaat, Tay & de Barros изучала effects of street pattern, traffic, road infrastructure, socioeconomic and demographic characteristics on public transit ridership в 185 community areas. В аннотации цель обозначена как выявление эффектов разных street patterns на transit ridership.

Source:  
https://doi.org/10.1007/s12205-016-0693-6

В отчёте University of Manitoba пересказывается конкретный результат Pasha et al.:

> Warped parallel and mixed patterns are associated with an increase in transit ridership, while neither the curvilinear pattern nor the grid pattern showed any effect.

Source:  
https://umanitoba.ca/architecture/sites/architecture/files/2021-02/cp_2018-2019_capstone_metalnikov_report.pdf

Для вопроса про доступность до сервисов это можно использовать узко:

> Если сервис = PT stop / transit access, то street-pattern knowledge не сводится к бинарному “grid хорошо, cul-de-sac плохо”. Есть эмпирический результат, где warped parallel и mixed связаны с higher transit ridership, а grid и curvilinear не показали отдельного эффекта.

Важно:

> Этот источник не говорит автоматически, что warped parallel “лучше” для всех сервисов или всех городов.

---

## 5. Как применять это к задаче “попадать в заданный network-distance radius”

### Step 1. Считать service area только по network distance

Фактическая доступность определяется street/path network, а не straight-line radius.

Sources:

- https://humantransit.org/2010/05/culdesac-hell-and-the-radius-of-demand.html
- https://arxiv.org/abs/1708.00836

---

### Step 2. Найти адреса / ячейки внутри и вне порога

Нужно сравнивать:

- адреса / ячейки, которые попадают в заданный network-distance threshold;
- адреса / ячейки, которые не попадают;
- случаи, где straight-line proximity есть, но network distance превышает порог.

Именно такие случаи Human Transit показывает как проблему cul-de-sac / barrier / access path.

Source:  
https://humantransit.org/2010/05/culdesac-hell-and-the-radius-of-demand.html

---

### Step 3. Классифицировать street pattern / connectivity condition

Возможные категории:

- grid;
- cul-de-sac / curvilinear;
- mixed;
- warped parallel;
- superblock / large block;
- disconnected pattern.

Литература использует такие категории в анализе street patterns и transit ridership, включая Pasha et al. и пересказ University of Manitoba.

Sources:

- https://doi.org/10.1007/s12205-016-0693-6
- https://umanitoba.ca/architecture/sites/architecture/files/2021-02/cp_2018-2019_capstone_metalnikov_report.pdf

---

### Step 4. Для cul-de-sac / disconnected / large-block areas проверять pedestrian links

Проверять:

- pedestrian links;
- cycle links;
- cut-throughs;
- paths through parks / greenbelts;
- pedestrian links through large blocks;
- extensions from dead ends to nearby streets.

Это прямо предлагают Global Designing Cities Initiative, Sustainable City Code и UC Davis.

Sources:

- https://globaldesigningcities.org/publication/global-street-design-guide/designing-streets-people/designing-for-pedestrians/pedestrian-networks/
- https://sustainablecitycode.org/brief/pedestrian-connectivity-through-culs-de-sac/
- https://itspubs.ucdavis.edu/download_pdf.php?id=1665

---

### Step 5. Для grid-like areas не ожидать большого выигрыша от изменения самого street pattern

Grid обычно уже даёт более прямую pedestrian accessibility относительно cul-de-sac.

Если порог доступности не достигается в grid-like area, источники скорее направляют к проверке:

- расположения сервисов;
- барьеров;
- переходов;
- качества pedestrian network;
- плотности сервисной сети.

Source:  
https://wfrc.utah.gov/PublicInvolvement/InTheNews/AssessmentOfEffectsOfStreetConnectivity.pdf

---

### Step 6. Для PT отдельно учитывать warped parallel и mixed patterns

Для public transport relevant source:

Pasha et al. / University of Manitoba report:

> Warped parallel and mixed street patterns were associated with an increase in transit ridership, while grid and curvilinear patterns showed no effect.

Sources:

- https://doi.org/10.1007/s12205-016-0693-6
- https://umanitoba.ca/architecture/sites/architecture/files/2021-02/cp_2018-2019_capstone_metalnikov_report.pdf

Использование:

> Не ограничиваться сравнением grid vs cul-de-sac при анализе PT accessibility / ridership; отдельно выделять warped parallel и mixed patterns.

---

## 6. Короткая формулировка для текста

> В рамках network-distance accessibility street pattern используется как диагностический признак того, почему часть адресов не попадает в заданный порог доступности до сервиса. Источники подчёркивают, что фактическая walking catchment area определяется сетью улиц и путей, а не straight-line radius. Для cul-de-sac и disconnected patterns литература указывает на увеличение фактических walking distances и рекомендует повышать pedestrian connectivity через links, cut-throughs, paths через large blocks, parks и greenbelts. Для grid-like patterns эффект изменения самого паттерна менее очевиден, поскольку grid обычно уже обеспечивает более прямую pedestrian accessibility. Для public transport отдельно важна работа Pasha et al., где warped parallel и mixed street patterns были ассоциированы с увеличением transit ridership, тогда как grid и curvilinear patterns не показали отдельного эффекта.

Sources:

- https://humantransit.org/2010/05/culdesac-hell-and-the-radius-of-demand.html
- https://arxiv.org/abs/1708.00836
- https://globaldesigningcities.org/publication/global-street-design-guide/designing-streets-people/designing-for-pedestrians/pedestrian-networks/
- https://sustainablecitycode.org/brief/pedestrian-connectivity-through-culs-de-sac/
- https://itspubs.ucdavis.edu/download_pdf.php?id=1665
- https://wfrc.utah.gov/PublicInvolvement/InTheNews/AssessmentOfEffectsOfStreetConnectivity.pdf
- https://doi.org/10.1007/s12205-016-0693-6
- https://umanitoba.ca/architecture/sites/architecture/files/2021-02/cp_2018-2019_capstone_metalnikov_report.pdf