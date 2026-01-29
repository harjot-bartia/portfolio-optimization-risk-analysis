# portfolio-optimization-risk-analysis
A Python project that applies Modern Portfolio Theory to optimize asset allocations and analyze portfolio risk using historical stock market data.

The project downloads financial data, computes returns and volatility, generates an efficient frontier, and evaluates risk metrics such as Value at Risk (VaR), Conditional VaR (CVaR), and maximum drawdown.

---

## Features

- Historical stock price data using Yahoo Finance
- Portfolio return and volatility analysis
- Correlation matrix visualization
- Efficient frontier simulation (10,000 portfolios)
- Optimal portfolio selection (maximum Sharpe ratio)
- Risk metrics:
  - Value at Risk (VaR)
  - Conditional VaR (CVaR)
  - Maximum drawdown
- Automated report generation

---

## Technologies Used

- Python  
- NumPy  
- Pandas  
- SciPy  
- Matplotlib  
- Seaborn  
- yFinance  

---

## Project Structure
portfolio_optimization_project/
├── example_analysis.py
├── portfolio_optimizer.py
├── requirements.txt
├── README.md
├── outputs/
│ ├── correlation_matrix.png
│ ├── efficient_frontier.png
│ ├── drawdown_chart.png
│ ├── cumulative_returns.png
│ └── portfolio_report.txt


---

## How to Run

### 1. Clone the repository

```
git clone https://github.com/yourusername/portfolio_optimization_project.git
cd portfolio_optimization_project

2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux

3. Install dependencies
python -m pip install -r requirements.txt

4. Run the analysis
python example_analysis.py

Example Outputs
Efficient Frontier

Correlation Matrix

Drawdown Chart

Cumulative Returns


Key Results (Sample Run)

Optimal portfolio selected using maximum Sharpe ratio

Expected annual return: ~50%

Annual volatility: ~28%

Sharpe ratio: ~1.79

Maximum drawdown: ~28%

(Results vary based on time period and assets selected)


What I Learned

- Applying Modern Portfolio Theory in practice

- Portfolio optimization using numerical methods

- Risk measurement techniques in finance

- Working with real financial market data

- Data visualization for financial analysis


Future Improvements

- Add support for user input portfolios

- Include transaction costs

- Add Monte Carlo portfolio simulations

- Web or GUI interface'

Author

Harjot Bartia

