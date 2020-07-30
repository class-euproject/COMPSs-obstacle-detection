#!/bin/bash

runcompss --lang=python --project=config/project.xml --resources=config/resources.xml --cpu_affinity=disabled --python_interpreter=python3 --classpath=/home/eudald/scwc_demo/CLASSDemo/app/dataclay.jar --scheduler="es.bsc.compss.scheduler.paper.PaperScheduler" --scheduler_config_file=config/config.file app.py
