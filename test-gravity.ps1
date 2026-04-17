$body = @{
    patient_id = "test123"
    patient_name = "Test Patient"
    answers = @(
        @{question = "pain level"; answer = 7},
        @{question = "temperature"; answer = 38.5},
        @{question = "heart rate"; answer = 95},
        @{question = "oxygen saturation"; answer = 92},
        @{question = "blood pressure"; answer = "140/90"}
    )
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "https://gravity-service.onrender.com/predict-gravity" -Method Post -Body $body -ContentType "application/json"
