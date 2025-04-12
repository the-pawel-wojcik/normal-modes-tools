# Transformation of gradient 
Transform a gradient expressed in dimensionless normal coordinates from regular
masses to deuterated masses.

## Run the tests with
```bash
python deuterate_xsim_gradient.py\
    --gradient_fname inputs/sroph_at_g0_kappa_a.json | jq > test_kappa.json
nvim -d test_kappa.json outputs/sroph-5d_at_g0_kappa_a.json
python deuterate_xsim_gradient.py\
    --gradient_fname inputs/sroph_at_g0_lambda_a.json | jq > test_lambda.json
nvim -d test_lambda.json outputs/sroph-5d_at_g0_lambda_a.json
rm test_kappa.json test_lambda.sjon
```
