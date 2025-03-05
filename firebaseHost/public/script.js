const API_URL = "https://rain-d6m9.onrender.com/predict";

document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("predict-form").addEventListener("submit", async function (event) {
        event.preventDefault();

        const precipitation = document.getElementById("precipitation").value;
        const temp_max = document.getElementById("temp_max").value;
        const temp_min = document.getElementById("temp_min").value;
        const wind = document.getElementById("wind").value;

        if (!precipitation || !temp_max || !temp_min || !wind) {
            alert("กรุณากรอกข้อมูลให้ครบทุกช่อง!");
            return;
        }

        const data = {
            precipitation: parseFloat(precipitation),
            temp_max: parseFloat(temp_max),
            temp_min: parseFloat(temp_min),
            wind: parseFloat(wind)
        };

        console.log("Sending data:", JSON.stringify(data));

        try {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(`Server error: ${result.error}`);
            }

            console.log("Server Response:", result);

            
            const resultElement = document.getElementById("result");
            resultElement.innerText = `พยากรณ์สภาพอากาศ: ${result.prediction}`;
            resultElement.style.display = "block"; 
            resultElement.style.visibility = "visible";

            let textColorMap = {

                "rain": "#3366FF",
                "sun": "#FF6633",
                "snow": "#000000",
                "drizzle": "#00FF33",
                "fog": "#663300"
            }

            resultElement.style.color = textColorMap[result.prediction] || "000000"
            

        } catch (error) {
            console.error("Error:", error);
            const resultElement = document.getElementById("result");
            resultElement.innerText = `เกิดข้อผิดพลาด: ${error.message}`;
            resultElement.style.display = "block";
            resultElement.style.visibility = "visible";
        }
    });
});
