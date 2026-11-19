async function predictPrice() {

    const result = document.getElementById("result");

    result.innerHTML = "Predicting...";

    const data = {

        vehicle_age:
        document.getElementById("vehicle_age").value,

        km_driven:
        document.getElementById("km_driven").value,

        mileage:
        document.getElementById("mileage").value,

        engine:
        document.getElementById("engine").value,

        max_power:
        document.getElementById("max_power").value,

        seats:
        document.getElementById("seats").value,

        seller:
        document.getElementById("seller").value,

        fuel:
        document.getElementById("fuel").value,

        transmission:
        document.getElementById("transmission").value
    };

    const response = await fetch("/predict_price", {

        method:"POST",

        headers:{
            "Content-Type":"application/json"
        },

        body:JSON.stringify(data)
    });

    const resultData = await response.json();

    const price =
    Number(resultData.price).toLocaleString(
    "en-IN"
    );

    result.innerHTML =
    `₹ ${price}`;
}