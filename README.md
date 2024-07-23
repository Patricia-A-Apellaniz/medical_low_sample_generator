<!-- ABOUT THE PROJECT -->

## Synthetic Data Generator in Data Scarce Medical Environments


This repository provides:
* Necessary scripts to preprocess the data and apply the generation methodology.
* Functions to generate result tables and plots as presented in the paper. 
* Experiments already done to save you time.

For more details, see full paper [TBC]().


<!-- GETTING STARTED -->
## Getting Started
Follow these simple steps to make this project work on your local machine.

### Prerequisites
You should have the following installed on your machine:

* Ubuntu
* Python 3.9.0
* Packages in requirements.txt
  ```sh
  pip install -r requirements.txt
  ```

### Installation

Download the repo manually (as a .zip file) or clone it using Git.
   ```sh
   git clone https://github.com/Patricia-A-Apellaniz/medical_low_sample_generator
   ```


Already obtained table results can be found in /results/.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

[//]: # (Already trained models can be downloaded in the following link: [trained_models]&#40;&#41;)

You can specify different configurations or training parameters in main.py for every experiment. 

**Data Preprocessing**
  * Preprocess classification data. Run the following command:
   ```sh
   python data/main_preprocess_class_data.py
   ```
  * Preprocess survival analysis data. Run the following command:
   ```sh
   python data/main_preprocess_sa_data.py
   ```
**Data Generation and Validation with Scarce-Data Methodology**
 * Generate classification data. Run the following command:
     ```sh
     python main_class_data.py
     ```
 * Generate survival analysis data. Run the following command:
     ```sh
     python main_sa_data.py
    ``` 
**Additional Validation: Change Task**
* Classification data. Run the following command:
     ```sh
     python main_class_data_change_task.py
     ```
 * Survival analysis data. Run the following command:
     ```sh
     python main_sa_data_change_task.py
    ``` 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



[//]: # (<!-- LICENSE -->)

[//]: # (## License)

[//]: # ()
[//]: # (Distributed under the XXX License. See `LICENSE.txt` for more information.)

[//]: # ()
[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)



<!-- CONTACT -->
## Contact

Patricia A. Apellaniz - patricia.alonsod@upm.es

<p align="right">(<a href="#readme-top">back to top</a>)</p>


[//]: # (<!-- ACKNOWLEDGMENTS -->)

[//]: # (## Acknowledgments)

[//]: # ()
[//]: # (* []&#40;&#41;)

[//]: # (* []&#40;&#41;)

[//]: # (* []&#40;&#41;)

[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)