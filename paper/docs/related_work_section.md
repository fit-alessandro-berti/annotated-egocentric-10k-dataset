# Related Work

The present project is positioned at the intersection of process mining, event-log construction from unstructured visual observations, multimodal activity interpretation, and manufacturing process analytics. In contrast to the classical process-mining setting, where event logs are assumed to be available from an information system, this project derives event data from egocentric factory videos through a pipeline of video transcription, worker-level process summarization, factory-level activity vocabulary construction, and conversion into per-video and merged CSV event logs. The related work is therefore organized around the following themes: foundations of process mining, event-log construction, event-data quality and abstraction, sensor and video process mining, manufacturing applications, object-centric perspectives, LLM-assisted event extraction, and downstream process discovery and conformance analysis.

## Foundations of Process Mining and Event Data

Process mining connects observed execution data with process models and operational questions. In [AALST2012], process mining is presented as a bridge between data mining and process modeling, with discovery, conformance checking, and enhancement as core task families. The Process Mining Manifesto [MANIFESTO2012] further formalizes the field by emphasizing that process-mining results depend heavily on the availability, semantics, and quality of event data. The textbook [AALST2016] systematizes these ideas and defines the common event-log assumptions that remain central to most tools: events refer to cases, activities, timestamps, and optional resources or additional attributes.

This project adopts the same basic event-data view, but obtains the log from egocentric video observations rather than from a workflow, ERP, MES, or database system. The chapter [DEWEERDT2022] is therefore particularly relevant because it discusses the foundations of process event data, including how events, cases, activities, timestamps, resources, and attributes are represented and interpreted. The 360-degree overview in [AALST2022] also helps situate the work as more than a log-mining exercise: the goal is not merely to produce a table of events, but to enable process understanding, diagnosis, comparison, and improvement.

Project methodology is another important foundation. The PM2 methodology in [VANECK2015] structures process-mining projects around planning, extraction, data processing, mining, evaluation, and process improvement. The current project follows a similar logic, but expands the extraction and processing phases: raw videos are first translated into textual observations, then into worker and factory summaries, and finally into event logs. As observed in [AALST2024], process mining is increasingly moving beyond simple workflow-centric assumptions toward broader forms of behavior analysis that include multi-actor, multi-object, and operational contexts. That observation is directly relevant to egocentric factory work, where activities may involve workers, tools, materials, machines, bins, workstations, and intermediate products.

Standardized event-data exchange has also shaped the field. The XES standard and its adoption are discussed in [WYNN2024], while many tools still support CSV-style logs for practical analysis. The event logs in this project are CSV-based and include start and end timestamps, activity, process, factory, worker, source file, case identifier, and duration. This makes the generated logs accessible to common process-mining workflows, while leaving room for later conversion to richer standards such as XES or OCEL.

## Event-Log Construction and Pre-Analysis

Event-log construction is a central related-work stream because this project’s main contribution lies before classical process mining begins. In [AALST2015], van der Aalst discusses how event data can be extracted from databases to unleash process mining. That work focuses on structured data, yet it highlights a general problem that also appears here: process mining requires deliberate choices about what counts as an event, which event attributes to retain, how to define cases, and how to map low-level observations to process-level activities.

The paper [DIBA2020] provides a useful conceptual decomposition of event-data preparation into extraction, correlation, and abstraction. In the present project, extraction corresponds to identifying process-relevant observations from video-derived transcripts, correlation corresponds to assigning events to a worker, video, factory, process, and case, and abstraction corresponds to mapping detailed observations into controlled factory activity labels. Similarly, [DEMURILLAS2020] studies automated case notion discovery and recommendation for database-derived logs. Although the data source differs, the case-notion problem remains: the project can treat a single video, a worker sequence, a factory-level process, a product journey, or a material-handling episode as a case, and each choice supports a different analysis.

Quality-informed event-log generation is addressed in [ANDREWS2020], which argues that event-log construction should make quality concerns explicit rather than treating log creation as a purely technical conversion step. This is highly relevant for LLM-generated logs, where timestamps, activity labels, process labels, and event boundaries are inferred rather than natively recorded. The review [PRADHAN2025] broadens this perspective by analyzing the pre-analysis stage of process mining, including data extraction, transformation, preparation, and quality assessment. The current project can be framed as a pre-analysis pipeline for an otherwise unavailable class of event data: manual, visual, worker-centered factory behavior.

A key implication of this literature is that the event log is not a neutral representation of the videos. Each event is the result of modeling decisions: how granular an activity should be, when an activity starts and ends, whether waiting counts as an event, how repeated micro-actions are aggregated, and how uncertain observations are handled. The project’s controlled activity vocabulary is therefore not only an implementation detail; it is an event-log construction mechanism that determines what process-mining questions the resulting log can answer.

## Event-Data Quality, Imperfections, and Abstraction

The generated logs in this project inherit uncertainty from several stages: visual ambiguity in egocentric videos, natural-language transcription, LLM interpretation, activity-vocabulary matching, and timestamp estimation. The event-log imperfection patterns described in [SURIADI2017] provide a vocabulary for such issues, including missing, incorrect, imprecise, and irrelevant event data. For this project, likely imperfections include imprecise start and end times, ambiguous activity boundaries, inconsistent labeling of similar manual actions, missing events during occlusions, and over-segmentation or under-segmentation of repetitive tasks.

The research agenda in [TERHOFSTEDE2023] argues that process-data quality is a frontier problem for process mining. This is especially important for derived logs, because errors are not merely formatting errors; they can change the discovered process model, the perceived bottlenecks, and the interpretation of worker behavior. In [GOEL2022], quality-informed process mining is supported through standardized data-quality annotations. A similar idea is useful here: each generated event could be enriched with provenance, confidence, prompt version, source transcript excerpt, vocabulary version, and whether the timestamp was observed, inferred, or interpolated.

Event abstraction is another key concern. The taxonomy in [VANZELST2021] reviews how low-level events can be lifted to higher-level process activities. The project performs a form of abstraction when it maps detailed video descriptions such as grasping a part, aligning it, pressing it into place, inspecting it, and putting it aside into broader activity labels such as assembling, inspecting, handling material, or operating a machine. Earlier work on activity mining by global trace segmentation [GUNTHER2010] and event abstraction for daily-life process descriptions [TAX2018] is relevant because egocentric video logs can contain continuous streams of fine-grained human actions rather than discrete system transactions.

This literature suggests that the paper should explicitly describe the abstraction level chosen for the activity vocabulary. A very fine-grained log may capture manual work accurately but produce unreadable spaghetti models; a very coarse log may support clean process discovery but hide relevant micro-variations, rework, waiting, and batching. The project’s factory-level activity catalogue can therefore be presented as a domain-specific abstraction layer designed to balance process-mining interpretability with fidelity to the original video observations.

## Sensor, Video, and Multimodal Process Mining

Sensor-based process mining is the closest established stream to video-derived event logs. In [MANNHARDT2018], Mannhardt and colleagues propose a taxonomy for combining activity recognition and process discovery in industrial environments. The framework in [KOSCHMIDER2020] and the work on process model discovery from sensor event data in [JANSSEN2021] show that process mining can be applied even when events must be inferred from raw signals rather than directly recorded by enterprise systems. The review [BRZYCHCZY2025] synthesizes process mining on sensor data and shows that the field has matured into a distinct research area, while also identifying continuing challenges around event detection, abstraction, evaluation, and contextualization.

Video process mining is even more directly related. The paper [KNOCH2020] proposes extracting traces from videos for process discovery and conformance checking in manual assembly. The reference architecture in [KRATSCH2022] argues that video can reduce blind spots caused by manual activities that are not captured by information systems. The analytics pipeline in [LEPSIEN2023] further develops the idea of process mining on video data, while [CHEN2023] explores video process mining and model matching for conformance checking. These papers establish that video can be used as a process-mining data source and that computer vision can support event extraction from manual work.

The current project differs from much of this work by using egocentric factory videos and a text-centered LLM pipeline rather than relying only on fixed-camera computer vision. Egocentric video captures what a worker sees and does, which can reveal fine-grained hand operations, material selection, tool use, movement between stations, and waiting. At the same time, egocentric video introduces challenges such as motion blur, occlusion by hands or tools, limited scene context, and variation in camera viewpoint. The LLM-mediated transcription step provides a flexible bridge from visual observations to process language, but it also introduces new quality and reproducibility questions.

Recent work on unstructured and multimodal process mining helps frame this contribution. The systematic literature review [KONIG2025] discusses process mining with unstructured data such as text, images, and video. In [GAVRIC2024], multimodal process mining is proposed as a way to combine heterogeneous data types for richer process analysis. The paper [STURM2026] instantiates a reference architecture for multimodal event-log construction and is particularly relevant because it treats event-log construction as an architecture that integrates multiple data sources and transformation functions. The present project can be positioned as a concrete instantiation for egocentric manufacturing video, where the modalities are video, generated text, structured activity vocabularies, and event-log tables.

## Process Mining in Manufacturing and Human-Centered Industrial Work

Manufacturing has long been a promising application domain for process mining because production systems generate operational data and require continuous improvement. The systematic review [CORALLO2020] surveys industrial applications of process mining and shows the relevance of process mining beyond administrative workflows. In [DREHER2021], manufacturing-specific application fields and research gaps are discussed, including the difficulty of obtaining suitable event data across heterogeneous production environments. The review [AKHRAMOVICH2024] connects process mining with Industry 4.0, where cyber-physical systems, automation, and human-machine cooperation create both richer data sources and more complex analysis problems.

Real-world manufacturing process mining often has to integrate multiple levels of behavior. The paper [BIRK2021] presents process mining for multi-level interlinked manufacturing processes, which is relevant because factory work can involve business-level orders, machine-level operations, worker-level actions, and material-level flows. The present project focuses especially on the worker-level and visual level, where many important actions are not recorded in production systems: picking parts, orienting components, operating simple machines, waiting for a station, transferring materials, inspecting outcomes, correcting defects, and moving between locations.

Human-centered industrial process mining also raises privacy, ethics, and interpretation issues. In [MANNHARDT2018PRIVACY], privacy challenges are discussed for process mining in human-centered industrial environments. Egocentric video intensifies these concerns because it may capture workers, bystanders, work practices, physical layouts, and potentially sensitive production details. A related-work section for this project should therefore not treat video-derived event logs only as a technical achievement. It should also acknowledge that worker-centered logs require careful anonymization, aggregation, access control, and purpose limitation, especially when the logs could be used for performance comparison or surveillance.

For manufacturing analysis, the project opens a complementary perspective to ERP, MES, and sensor logs. Instead of only tracking orders or machine states, it can reveal how manual operations unfold in practice. This supports process questions such as which activities dominate a worker’s time, where waiting and walking occur, how material replenishment interacts with production work, how workers deviate from expected sequences, and whether similar factories or workers follow different routines.

## Object-Centric and Multi-Perspective Process Mining

Most classical process-mining techniques assume a single case notion. In this project, the initial CSV logs use a case-centric structure, for example by associating events with videos, workers, and factories. However, factory work is naturally multi-object: one event may involve a worker, a workstation, a tool, a machine, a bin, a component, a batch, and a partially assembled product. This creates the divergence and convergence problems described in [AALST2019OC], where flattening multi-object behavior into one case notion can duplicate events, hide synchronization, or create misleading loops.

Object-centric process mining is therefore an important direction for extending the project. The paper [AALSTBERTI2020] introduces object-centric Petri nets as a modeling formalism for behavior involving multiple object types. The OCEL standard in [GHAHFAROKHI2021] provides an event-log format for object-centric event data, and the OCEL 2.0 specification [BERTI2024OCEL] extends the representation of objects, object relationships, event-object relationships, and attribute changes. These ideas are directly applicable if the project moves from worker/video-level cases to events that reference specific machines, material containers, product units, or workstations.

Event knowledge graphs provide a related multi-perspective approach. In [FAHLAND2022], process mining over multiple behavioral dimensions is discussed using event knowledge graphs. This is relevant because egocentric video observations can contain rich relational information that is lost in a flat table: a worker takes an item from a bin, places it on a fixture, actuates a machine, inspects the result, and transfers it to another location. Representing these observations as graph relations could preserve more context and later support both object-centric process mining and semantic querying.

For the current paper, object-centric process mining can be used to motivate future work without requiring that the present dataset already be fully object-centric. The generated CSV logs are a practical first layer for conventional discovery and performance analysis, while OCEL or event-knowledge-graph representations would enable richer analyses of material flow, resource sharing, batching, and interactions between workers and machines.

## LLM-Assisted Process Mining and Event Extraction

Large language models are increasingly discussed in process mining as tools for querying, abstraction, explanation, and event-log construction. In [BERTI2024LLM], Berti, Schuster, and van der Aalst study abstractions, scenarios, and prompt definitions for process mining with LLMs, showing how process-mining artifacts can be represented in prompts and queried through natural language. The benchmark in [BERTI2025BENCH] evaluates LLMs on process-mining tasks and highlights both their potential and the need for careful evaluation. The PM4Py.LLM module described in [BERTI2024PM4PYLLM] further indicates that LLM-based interaction is becoming part of practical process-mining toolchains.

LLMs are also being applied to the upstream problem of event-log extraction. In [DANI2025], event logs are extracted for process mining using large language models, particularly in settings where manually writing extraction logic is time-consuming. The Text2EL+ framework in [GEEGANAGE2024] enriches event logs using unstructured text with expert guidance. The paper [KECHT2021] extracts event logs from customer-service conversations using natural language inference. These works are relevant because the current project also transforms unstructured or semi-structured natural-language descriptions into process-mining event data.

The project differs by applying LLMs after a multimodal-to-text transformation from egocentric video. Rather than only converting a database schema, document collection, or conversation transcript into an event log, the pipeline first creates detailed chronological video descriptions and then converts them into events under a controlled activity and process vocabulary. This makes prompt design and domain constraints central to the contribution. The activity vocabulary can reduce hallucinated labels, improve comparability across workers and factories, and make the resulting log easier to mine. At the same time, LLM outputs require validation because plausible event labels may still be incorrect, missing, temporally imprecise, or inconsistent across videos.

This literature suggests several evaluation dimensions for the paper. The generated logs should be evaluated not only by whether they parse as CSV, but also by event boundary accuracy, activity-label accuracy, process-label accuracy, timestamp error, case assignment quality, consistency across similar videos, and usefulness for downstream process-mining tasks. Human validation on a sample of videos, comparison against manual annotations, prompt-version tracking, and ablation studies with and without controlled vocabularies would all align the project with concerns raised in LLM-based process-mining research.

## Process Discovery, Conformance Checking, and Tool Support

Once event logs are available, classical process-mining techniques become applicable. The Inductive Miner in [LEEMANS2013] is relevant because it discovers structured process models that are often easier to interpret than highly unstructured models. The Flexible Heuristics Miner in [WEIJTERS2011] and the Fuzzy Miner in [GUNTHER2007] are also relevant for noisy or complex event logs, where frequency and significance thresholds can simplify behavior. Split Miner [AUGUSTO2019] addresses the trade-off between accuracy, simplicity, and generalization, which is especially important for video-derived logs that may contain both noise and genuine behavioral variation.

Conformance checking is relevant if the project compares observed worker behavior against expected factory procedures, standard operating procedures, or discovered reference models. The replay-based perspective in [AALST2012REPLAY] connects conformance checking with performance analysis by replaying event history on process models. For this project, conformance analysis could help identify missing steps, unexpected rework, alternative task sequences, or deviations between workers and factories. It could also be used cautiously to compare observed manual behavior against generated or expert-defined process models.

Tool support matters because the project outputs practical event logs. PM4Py [BERTI2023PM4PY] provides a Python-based library for process mining, including event-log manipulation, discovery algorithms, conformance checking, and performance analysis. Because the project itself is implemented as a Python pipeline and produces CSV event logs, PM4Py is a natural tool for demonstrating the usefulness of the generated data. A paper can therefore include downstream analyses such as directly-follows graphs, process variants, activity frequencies, duration distributions, worker comparisons, factory-level process maps, and bottleneck detection.

The discovery and conformance literature also cautions that downstream results should not be interpreted independently of log-generation quality. If activity labels are too granular, discovery may produce complex models; if labels are too coarse, conformance analysis may miss relevant deviations; if timestamps are estimated, performance analysis should report uncertainty. Therefore, the generated event logs should be presented as a research artifact whose validity depends on both the video-to-event pipeline and the process-mining methods applied afterward.

## Positioning of This Work

The closest related work combines three lines of research. The first is event-log construction and pre-analysis, represented by [AALST2015], [DIBA2020], [ANDREWS2020], and [PRADHAN2025]. These works motivate the need to treat event-log creation as a methodological and quality-sensitive process rather than a simple export operation. The second is video, sensor, and multimodal process mining, represented by [MANNHARDT2018], [KNOCH2020], [KRATSCH2022], [LEPSIEN2023], [KONIG2025], [GAVRIC2024], and [STURM2026]. These works motivate the use of non-traditional data sources to observe manual and physical work. The third is LLM-assisted process mining and event extraction, represented by [BERTI2024LLM], [BERTI2025BENCH], [DANI2025], [GEEGANAGE2024], and [KECHT2021]. These works motivate the use of language models to transform unstructured descriptions into process-mining artifacts.

Against this background, the project contributes a concrete pipeline for deriving process-mining event logs from egocentric factory videos. Its distinctive focus is the combination of worker-centered visual observations, LLM-generated chronological transcriptions, factory-specific activity vocabularies, and conventional event-log outputs that can be analyzed with existing process-mining tools. The work therefore addresses a blind spot in process mining: manual factory activities that are visible in video but absent from enterprise systems.

The related literature also clarifies the main limitations that the paper should acknowledge. The generated log is an inferred representation, not a direct recording of system transactions. Event boundaries, timestamps, activity labels, and process labels may be uncertain. A flat CSV log may not preserve all worker-object-machine-material relations. Privacy and ethical constraints are central when using worker-centered video. These limitations do not undermine the contribution; rather, they define the research agenda for evaluating and extending the dataset, for example through quality annotations, human validation, object-centric representations, and multimodal fusion with other factory data sources.

REFERENCES

[AALST2012] Wil M. P. van der Aalst. “Process Mining: Overview and Opportunities.” ACM Transactions on Management Information Systems, 3(2), Article 7, 2012. DOI: 10.1145/2229156.2229157.

[MANIFESTO2012] Wil M. P. van der Aalst et al. “Process Mining Manifesto.” In Business Process Management Workshops, LNBIP 99, pp. 169–194, Springer, 2012. DOI: 10.1007/978-3-642-28108-2_19.

[AALST2016] Wil M. P. van der Aalst. Process Mining: Data Science in Action. 2nd ed., Springer, 2016. DOI: 10.1007/978-3-662-49851-4.

[DEWEERDT2022] Jochen De Weerdt and Moe Thandar Wynn. “Foundations of Process Event Data.” In Wil M. P. van der Aalst and Josep Carmona, editors, Process Mining Handbook, LNBIP 448, pp. 193–211, Springer, 2022. DOI: 10.1007/978-3-031-08848-3_6.

[AALST2022] Wil M. P. van der Aalst. “Process Mining: A 360 Degree Overview.” In Wil M. P. van der Aalst and Josep Carmona, editors, Process Mining Handbook, LNBIP 448, pp. 3–34, Springer, 2022. DOI: 10.1007/978-3-031-08848-3_1.

[VANECK2015] Maikel L. van Eck, Xixi Lu, Sander J. J. Leemans, and Wil M. P. van der Aalst. “PM2: A Process Mining Project Methodology.” In Advanced Information Systems Engineering, CAiSE 2015, LNCS 9097, pp. 297–313, Springer, 2015. DOI: 10.1007/978-3-319-19069-3_19.

[AALST2024] Wil M. P. van der Aalst, Hajo A. Reijers, and Laura Maruster. “Process Mining Beyond Workflows.” Computers in Industry, 161, Article 104126, 2024. DOI: 10.1016/j.compind.2024.104126.

[WYNN2024] Moe T. Wynn, Wil M. P. van der Aalst, Eric Verbeek, and Bruno N. Di Stefano. “The IEEE XES Standard for Process Mining: Experiences, Adoption, and Revision.” IEEE Computational Intelligence Magazine, 19(1), pp. 20–23, 2024.

[AALST2015] Wil M. P. van der Aalst. “Extracting Event Data from Databases to Unleash Process Mining.” In BPM — Driving Innovation in a Digital World, pp. 105–128, Springer, 2015. DOI: 10.1007/978-3-319-14430-6_8.

[DIBA2020] Kiarash Diba, Kimon Batoulis, Matthias Weidlich, and Mathias Weske. “Extraction, Correlation, and Abstraction of Event Data for Process Mining.” WIREs Data Mining and Knowledge Discovery, 10(3), Article e1346, 2020. DOI: 10.1002/widm.1346.

[DEMURILLAS2020] Eduardo González López de Murillas, Hajo A. Reijers, and Wil M. P. van der Aalst. “Case Notion Discovery and Recommendation: Automated Event Log Building on Databases.” Knowledge and Information Systems, 62(7), pp. 2539–2575, 2020. DOI: 10.1007/s10115-019-01430-6.

[ANDREWS2020] Robert Andrews, Christopher G. J. van Dun, Moe T. Wynn, Wolfgang Kratsch, Maximilian Röglinger, and Arthur H. M. ter Hofstede. “Quality-Informed Semi-Automated Event Log Generation for Process Mining.” Decision Support Systems, 132, Article 113265, 2020. DOI: 10.1016/j.dss.2020.113265.

[PRADHAN2025] Shameer K. Pradhan, Mieke Jans, and Niels Martin. “Getting the Data in Shape for Your Process Mining Analysis: An In-Depth Analysis of the Pre-Analysis Stage.” ACM Computing Surveys, 57(6), Article 159, pp. 1–37, 2025. DOI: 10.1145/3712587.

[SURIADI2017] S. Suriadi, Robert Andrews, Arthur H. M. ter Hofstede, and Moe T. Wynn. “Event Log Imperfection Patterns for Process Mining: Towards a Systematic Approach to Cleaning Event Logs.” Information Systems, 64, pp. 132–150, 2017. DOI: 10.1016/j.is.2016.07.011.

[TERHOFSTEDE2023] Arthur H. M. ter Hofstede, Agnes Koschmider, Andrea Marrella, Robert Andrews, Dominik A. Fischer, Sareh Sadeghianasl, Moe Thandar Wynn, Marco Comuzzi, Jochen De Weerdt, Kanika Goel, Niels Martin, and Pnina Soffer. “Process-Data Quality: The True Frontier of Process Mining.” ACM Journal of Data and Information Quality, 15(3), Article 29, pp. 1–21, 2023. DOI: 10.1145/3613247.

[GOEL2022] Kanika Goel, Sander J. J. Leemans, Niels Martin, and Moe Thandar Wynn. “Quality-Informed Process Mining: A Case for Standardised Data Quality Annotations.” ACM Transactions on Knowledge Discovery from Data, 16(5), Article 97, pp. 1–47, 2022. DOI: 10.1145/3511707.

[VANZELST2021] Sebastiaan J. van Zelst, Felix Mannhardt, Massimiliano de Leoni, and Agnes Koschmider. “Event Abstraction in Process Mining: Literature Review and Taxonomy.” Granular Computing, 6(3), pp. 719–736, 2021. DOI: 10.1007/s41066-020-00226-2.

[GUNTHER2010] Christian W. Günther, Anne Rozinat, and Wil M. P. van der Aalst. “Activity Mining by Global Trace Segmentation.” In Business Process Management Workshops, LNBIP 43, pp. 128–139, Springer, 2010. DOI: 10.1007/978-3-642-12186-9_13.

[TAX2018] Niek Tax, Natalia Sidorova, Reinder Haakma, and Wil M. P. van der Aalst. “Mining Process Model Descriptions of Daily Life Through Event Abstraction.” In Intelligent Systems and Applications, SCI 751, pp. 83–104, Springer, 2018. DOI: 10.1007/978-3-319-69266-1_5.

[MANNHARDT2018] Felix Mannhardt, Riccardo Bovo, Manuel Fradinho Oliveira, and Simon Julier. “A Taxonomy for Combining Activity Recognition and Process Discovery in Industrial Environments.” In Intelligent Data Engineering and Automated Learning — IDEAL 2018, LNCS 11315, pp. 84–93, Springer, 2018. DOI: 10.1007/978-3-030-03496-2_10.

[KOSCHMIDER2020] Agnes Koschmider, Dominik Janssen, and Felix Mannhardt. “Framework for Process Discovery from Sensor Data.” In Proceedings of EMISA 2020, CEUR Workshop Proceedings, Vol. 2628, pp. 32–38, 2020.

[JANSSEN2021] Dominik Janssen, Felix Mannhardt, Agnes Koschmider, and Sebastiaan J. van Zelst. “Process Model Discovery from Sensor Event Data.” In Process Mining Workshops, ICPM 2020, LNBIP 406, pp. 69–81, Springer, 2021. DOI: 10.1007/978-3-030-72693-5_6.

[BRZYCHCZY2025] Edyta Brzychczy, Milda Aleknonytė-Resch, Dominik Janssen, and Agnes Koschmider. “Process Mining on Sensor Data: A Review of Related Works.” Knowledge and Information Systems, 67(6), pp. 4915–4948, 2025. DOI: 10.1007/s10115-024-02297-y.

[KNOCH2020] Sönke Knoch, Shreeraman Ponpathirkoottam, and Tim Schwartz. “Video-to-Model: Unsupervised Trace Extraction from Videos for Process Discovery and Conformance Checking in Manual Assembly.” In Business Process Management, BPM 2020, LNCS 12168, pp. 291–308, Springer, 2020. DOI: 10.1007/978-3-030-58666-9_17.

[KRATSCH2022] Wolfgang Kratsch, Fabian König, and Maximilian Röglinger. “Shedding Light on Blind Spots: Developing a Reference Architecture to Leverage Video Data for Process Mining.” Decision Support Systems, 158, Article 113794, 2022. DOI: 10.1016/j.dss.2022.113794.

[LEPSIEN2023] Arvid Lepsien, Agnes Koschmider, and Wolfgang Kratsch. “Analytics Pipeline for Process Mining on Video Data.” In Business Process Management Forum, LNBIP, pp. 196–213, Springer, 2023. DOI: 10.1007/978-3-031-41623-1_12.

[CHEN2023] Shuang Chen, Minghao Zou, Rui Cao, Ziqi Zhao, and Qingtian Zeng. “Video Process Mining and Model Matching for Intelligent Development: Conformance Checking.” Sensors, 23(8), Article 3812, 2023. DOI: 10.3390/s23083812.

[KONIG2025] Fabian König, Andreas Egger, Wolfgang Kratsch, Maximilian Röglinger, and Niklas Wördehoff. “Unstructured Data in Process Mining: A Systematic Literature Review.” ACM Transactions on Management Information Systems, 16(3), Article 25, pp. 1–34, 2025. DOI: 10.1145/3727148.

[GAVRIC2024] Aleksandar Gavric, Dominik Bork, and Henderik A. Proper. “Multimodal Process Mining.” In 2024 IEEE 26th Conference on Business Informatics, pp. 99–108, 2024. DOI: 10.1109/CBI62504.2024.00021.

[STURM2026] Frank Sturm, Annina Liessmann, and Martin Matzner. “Multimodal Event Log Construction for Process Mining: Instantiating a Reference Architecture.” In Design for Better Futures: Beyond the Science of the Artificial, DESRIST 2026, LNCS 16605, pp. 146–164, Springer, 2026. DOI: 10.1007/978-3-032-28316-0_9.

[CORALLO2020] Angelo Corallo, Mariangela Lazoi, and Fabrizio Striani. “Process Mining and Industrial Applications: A Systematic Literature Review.” Knowledge and Process Management, 27(3), pp. 225–233, 2020. DOI: 10.1002/kpm.1630.

[DREHER2021] Simon Dreher, Peter Reimann, and Christoph Gröger. “Application Fields and Research Gaps of Process Mining in Manufacturing Companies.” In INFORMATIK 2020, pp. 621–634, Gesellschaft für Informatik, 2021. DOI: 10.18420/inf2020_55.

[AKHRAMOVICH2024] Katsiaryna Akhramovich, Estefanía Serral, and Carlos Cetina. “A Systematic Literature Review on the Application of Process Mining to Industry 4.0.” Knowledge and Information Systems, 66, 2024. DOI: 10.1007/s10115-023-02042-x.

[BIRK2021] Alexander Birk, Yannick Wilhelm, Simon Dreher, Christian Flack, Peter Reimann, and Christoph Gröger. “A Real-World Application of Process Mining for Data-Driven Analysis of Multi-Level Interlinked Manufacturing Processes.” Procedia CIRP, 104, pp. 417–422, 2021. DOI: 10.1016/j.procir.2021.11.070.

[MANNHARDT2018PRIVACY] Felix Mannhardt, Sobah Abbas Petersen, and Manuel Fradinho Oliveira. “Privacy Challenges for Process Mining in Human-Centered Industrial Environments.” In 2018 14th International Conference on Intelligent Environments, pp. 64–71, IEEE, 2018. DOI: 10.1109/IE.2018.00017.

[AALST2019OC] Wil M. P. van der Aalst. “Object-Centric Process Mining: Dealing with Divergence and Convergence in Event Data.” In Software Engineering and Formal Methods, SEFM 2019, LNCS 11724, pp. 3–25, Springer, 2019. DOI: 10.1007/978-3-030-30446-1_1.

[AALSTBERTI2020] Wil M. P. van der Aalst and Alessandro Berti. “Discovering Object-Centric Petri Nets.” Fundamenta Informaticae, 175(1–4), pp. 1–40, 2020. DOI: 10.3233/FI-2020-1946.

[GHAHFAROKHI2021] Anahita Farhang Ghahfarokhi, Gyunam Park, Alessandro Berti, and Wil M. P. van der Aalst. “OCEL: A Standard for Object-Centric Event Logs.” In New Trends in Database and Information Systems, CCIS 1450, pp. 169–175, Springer, 2021. DOI: 10.1007/978-3-030-85082-1_16.

[BERTI2024OCEL] Alessandro Berti, Istvan Koren, Jan Niklas Adams, Gyunam Park, Benedikt Knopp, Nina Graves, Majid Rafiei, Lukas Liß, Leah Tacke genannt Unterberg, Yisong Zhang, Christopher Schwanen, Marco Pegoraro, and Wil M. P. van der Aalst. “OCEL (Object-Centric Event Log) 2.0 Specification.” arXiv:2403.01975, 2024.

[FAHLAND2022] Dirk Fahland. “Process Mining over Multiple Behavioral Dimensions with Event Knowledge Graphs.” In Wil M. P. van der Aalst and Josep Carmona, editors, Process Mining Handbook, LNBIP 448, pp. 274–319, Springer, 2022. DOI: 10.1007/978-3-031-08848-3_9.

[BERTI2024LLM] Alessandro Berti, Daniel Schuster, and Wil M. P. van der Aalst. “Abstractions, Scenarios, and Prompt Definitions for Process Mining with LLMs: A Case Study.” In Business Process Management Workshops, LNBIP, pp. 427–439, Springer, 2024. DOI: 10.1007/978-3-031-50974-2_32.

[BERTI2025BENCH] Alessandro Berti, Humam Kourani, and Wil M. P. van der Aalst. “PM-LLM-Benchmark: Evaluating Large Language Models on Process Mining Tasks.” In Process Mining Workshops, ICPM 2024, LNBIP 533, pp. 610–623, Springer, 2025. DOI: 10.1007/978-3-031-82225-4_45.

[BERTI2024PM4PYLLM] Alessandro Berti. “PM4Py.LLM: A Comprehensive Module for Implementing PM on LLMs.” arXiv:2404.06035, 2024.

[DANI2025] Vinicius Stein Dani, Marcus Dees, Henrik Leopold, Kiran Busch, Iris Beerepoot, Jan Martijn E. M. van der Werf, and Hajo A. Reijers. “Event Log Extraction for Process Mining Using Large Language Models.” In Cooperative Information Systems, CoopIS 2024, LNCS 15506, pp. 56–72, Springer, 2025. DOI: 10.1007/978-3-031-81375-7_4.

[GEEGANAGE2024] Dakshi Tharanga Kapugama Geeganage, Moe Thandar Wynn, and Arthur H. M. ter Hofstede. “Text2EL+: Expert Guided Event Log Enrichment Using Unstructured Text.” ACM Journal of Data and Information Quality, 16(1), Article 8, pp. 1–28, 2024. DOI: 10.1145/3640018.

[KECHT2021] Christoph Kecht, Andreas Egger, Wolfgang Kratsch, and Maximilian Röglinger. “Event Log Construction from Customer Service Conversations Using Natural Language Inference.” In 2021 3rd International Conference on Process Mining, pp. 144–151, IEEE, 2021. DOI: 10.1109/ICPM53251.2021.9576869.

[LEEMANS2013] Sander J. J. Leemans, Dirk Fahland, and Wil M. P. van der Aalst. “Discovering Block-Structured Process Models from Event Logs: A Constructive Approach.” In Application and Theory of Petri Nets and Concurrency, LNCS 7927, pp. 311–329, Springer, 2013. DOI: 10.1007/978-3-642-38697-8_17.

[WEIJTERS2011] A. J. M. M. Weijters and J. T. S. Ribeiro. “Flexible Heuristics Miner (FHM).” In IEEE Symposium on Computational Intelligence and Data Mining, pp. 310–317, IEEE, 2011. DOI: 10.1109/CIDM.2011.5949453.

[GUNTHER2007] Christian W. Günther and Wil M. P. van der Aalst. “Fuzzy Mining: Adaptive Process Simplification Based on Multi-Perspective Metrics.” In Business Process Management, BPM 2007, LNCS 4714, pp. 328–343, Springer, 2007. DOI: 10.1007/978-3-540-75183-0_24.

[AUGUSTO2019] Adriano Augusto, Raffaele Conforti, Marlon Dumas, Marcello La Rosa, and Artem Polyvyanyy. “Split Miner: Automated Discovery of Accurate and Simple Business Process Models from Event Logs.” Knowledge and Information Systems, 59(2), pp. 251–284, 2019. DOI: 10.1007/s10115-018-1214-x.

[AALST2012REPLAY] Wil M. P. van der Aalst, Arya Adriansyah, and Boudewijn F. van Dongen. “Replaying History on Process Models for Conformance Checking and Performance Analysis.” WIREs Data Mining and Knowledge Discovery, 2(2), pp. 182–192, 2012. DOI: 10.1002/widm.1045.

[BERTI2023PM4PY] Alessandro Berti, Sebastiaan J. van Zelst, and Daniel Schuster. “PM4Py: A Process Mining Library for Python.” Software Impacts, 17, Article 100556, 2023. DOI: 10.1016/j.simpa.2023.100556.
