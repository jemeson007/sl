import numpy as nyp

fllbck = {
"user_id": "12345",
"metrics": [
  {
   "date": "2024-11-22",
    "steps": 8500,
    "heart_rate": 75,
    "sleep_hours": 6.5,
    "hrv": 45
   },
  {
  "date": "2024-11-21",
  "steps": 9500,
  "heart_rate": 72,
  "sleep_hours": 7.2,
  "hrv": 50
  }
 ]
}

ff = fllbck.to_list()

print("variance", nyp.var(ff))