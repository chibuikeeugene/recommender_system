# SERVICE RECOMMENDER SYSTEM PACKAGE

A machine-based recommendation engine package that delivers personalized product recommendations more effectively across its entire customer base. This program uses an **svd algorithm from the surprise library** to personalize service offerings to account owners in a Spanish Santander bank.

## Data set and its description  

| Features                 | Description                                                                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| - fecha_dato             | - The table is partitioned for this column                                                                                                    |
| - ncodpers               | - Customer code                                                                                                                               |
| - ind_empleado           | - Employee index: A active, B ex employed, F filial, N not employee, P pasive                                                                 |
| - pais_residencia        | - Customer's Country residence                                                                                                                |
| - sexo                   | - Customer's sex                                                                                                                              |
| - age                    | - Age                                                                                                                                         |
| - fecha_alta             | - The date in which the customer became as the first holder of a contract in the bank                                                         |
| - ind_nuevo              | - New customer Index. 1 if the customer registered in the last 6 months.                                                                      |
| - antiguedad             | - Customer seniority (in months)                                                                                                              |
| - indrel                 | - 1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)                                                   |
| - ult_fec_cli_1t         | - Last date as primary customer (if he isn't at the end of the month)                                                                         |
| - indrel_1mes            | - Customer type at the beginning of the month ,1 (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner) |
| - tiprel_1mes            | - Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)                           |
| - indresi                | - Residence index (S (Yes) or N (No) if the residence country is the same than the bank country)                                              |
| - indext                 | - Foreigner index (S (Yes) or N (No) if the customer's birth country is different than the bank country)                                      |
| - conyuemp               | - Spouse index. 1 if the customer is spouse of an employee                                                                                    |
| - canal_entrada          | - channel used by the customer to join                                                                                                        |
| - indfall                | - Deceased index. N/S                                                                                                                         |
| - tipodom                | - Addres type. 1, primary address                                                                                                             |
| - cod_prov               | - Province code (customer's address)                                                                                                          |
| - nomprov                | - Province name                                                                                                                               |
| - ind_actividad_cliente  | - Activity index (1, active customer; 0, inactive customer)                                                                                   |
| - renta                  | - Gross income of the household                                                                                                               |
| - segmento               | - segmentation: 01 - VIP, 02 - Individuals 03 - college graduated                                                                             |
| - ind_ahor_fin_ult1      | - Saving Account                                                                                                                              |
| - ind_aval_fin_ult1      | - Guarantees                                                                                                                                  |
| - ind_cco_fin_ult1       | - Current Accounts                                                                                                                            |
| - ind_cder_fin_ult1      | - Derivada Account                                                                                                                            |
| - ind_cno_fin_ult1       | - Payroll Account                                                                                                                             |
| - ind_ctju_fin_ult1      | - Junior Account                                                                                                                              |
| - ind_ctma_fin_ult1      | - Más particular Account                                                                                                                      |
| - ind_ctop_fin_ult1      | - particular Account                                                                                                                          |
| - ind_ctpp_fin_ult1      | - particular Plus Account                                                                                                                     |
| - ind_deco_fin_ult1      | - Short-term deposits                                                                                                                         |
| - ind_deme_fin_ult1      | - Medium-term deposits                                                                                                                        |
| - ind_dela_fin_ult1      | - Long-term deposits                                                                                                                          |
| - ind_ecue_fin_ult1      | - e-account                                                                                                                                   |
| - ind_fond_fin_ult1      | - Funds                                                                                                                                       |
| - ind_hip_fin_ult1       | - Mortgage                                                                                                                                    |
| - ind_plan_fin_ult1      | - Pensions                                                                                                                                    |
| - ind_pres_fin_ult1      | - Loans                                                                                                                                       |
| - ind_reca_fin_ult1      | - Taxes                                                                                                                                       |
| - ind_tjcr_fin_ult1      | - Credit Card                                                                                                                                 |
| - ind_valo_fin_ult1      | - Securities                                                                                                                                  |
| - ind_viv_fin_ult1       | - Home Account                                                                                                                                |
| - ind_nomina_ult1        | - Payroll                                                                                                                                     |
| - ind_nom_pens_ult1      | - Pensions                                                                                                                                    |
| - ind_recibo_ult1        | - Direct Debit                                                                                                                                |
|                          |                                                                                                                                               |

## Dependencies and packages  

1. numpy>=1.20.0,<1.21.0
2. python = 3.10
3. pandas = 2.2.2
4. scikit-learn = 1.4.2
5. pydantic = 2.7.0
6. strictyaml = 1.7.3
7. tensorflow = 2.16.1
8. scikeras = 0.13.0
9. tensorflow-datasets = 4.9.4
10. pillow = 10.3.0
11. pydantic-settings = 2.2.1
12. fastapi = 0.110.3
13. uvicorn = 0.29.0
14. loguru = 0.7.2
15. python-multipart = 0.0.9
16. scikit-surprise = 1.1.4

## Source code link  

Source code link:
[Github link](https://github.com/chibuikeeugene/recommender_system.git)