import tf from "@tensorflow/tfjs-node";

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  // entrada
  model.add(
    tf.layers.dense({
      inputShape: 4,
      units: 300,
      activation: "relu",
    }),
  );

  // saida
  model.add(tf.layers.dense({ units: 5, activation: "softmax" }));

  // compilar
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // treinamento do modelo
  await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 600,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, log) =>
        console.log(`Epoch: ${epoch}: loss = ${log.loss}`),
    },
  });

  return model;
}

async function predict(model, paciente) {
  const tfInput = tf.tensor2d(paciente);

  const pred = model.predict(tfInput);
  const predArray = await pred.array();

  return predArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, )
const pessoas = [
  {
    // verde - doente mas pode aguardar
    nome: "João da Silva",
    idade: 10,
    freqCardiaca: 60,
    freqRespiratoria: 18,
    saturacao: 99,
  },
  {
    // vermelha - urgencia
    nome: "Maria Joaquina",
    idade: 12,
    freqCardiaca: 43,
    freqRespiratoria: 10,
    saturacao: 95,
  },
  {
    // verde - doente mas pode aguardar
    nome: "Peterson",
    idade: 8,
    freqCardiaca: 120,
    freqRespiratoria: 20,
    saturacao: 98,
  },
  {
    // vermelha - urgencia
    nome: "Joana",
    idade: 14,
    freqCardiaca: 170,
    freqRespiratoria: 30,
    saturacao: 80,
  },
  {
    // azul - pode esperar (nem era pra estar lá)
    nome: "Thomas",
    idade: 16,
    freqCardiaca: 80,
    freqRespiratoria: 17,
    saturacao: 100,
  },
  {
    // azul - pode espear (nem era pra estar lá)
    nome: "Antonela",
    idade: 9,
    freqCardiaca: 88,
    freqRespiratoria: 21,
    saturacao: 100,
  },
  {
    // amaerela - pouco pior que a verde
    nome: "José",
    idade: 5,
    freqCardiaca: 90,
    freqRespiratoria: 20,
    saturacao: 95,
  },
  {
    // laranja - pouco pior que a amarela (mas antes da vermelha)
    nome: "Jackson",
    idade: 7,
    freqCardiaca: 95,
    freqRespiratoria: 30,
    saturacao: 92,
  },
];

// ordem: [idade_normalizada, freqCardiaca_normalizada, freqRespiratoria_normalizada, saturacao_normalizada]
const tensorPacientesNormalizado = [
  [0.455, 0.134, 0.4, 0.95], // João da Silva
  [0.636, 0, 0, 0.75], // Maria Joaquina
  [0.273, 0.606, 0.5, 0.9], // Peterson
  [0.818, 1, 1, 0], // Joana
  [1, 0.291, 0.35, 1], // Thomas
  [0.364, 0.354, 0.55, 1], // Antonela
  [0, 0.37, 0.5, 0.75], // José
  [0.182, 0.409, 1, 0.6], // Jackson
];

// labels: [azul, verde, amarelo, laranja, vermelho]
const labels = ["azul", "verde", "amarelo", "laranja", "vermelho"];
const tensorLabels = [
  [0, 1, 0, 0, 0],
  [0, 0, 0, 0, 1],
  [0, 1, 0, 0, 0],
  [0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0],
  [1, 0, 0, 0, 0],
  [0, 0, 1, 0, 0],
  [0, 0, 0, 1, 0],
];

const inputXs = tf.tensor2d(tensorPacientesNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

const model = await trainModel(inputXs, outputYs);

/* NOVO PACIENTE */

const paciente = {
  nome: "Helena",
  idade: 5,
  freqCardiaca: 57,
  freqRespiratoria: 29,
  saturacao: 91,
};
const tensorPacienteNormalizado = [
  [0, 0.11, 0.95, 0.55], // Helena
];

const predictions = await predict(model, tensorPacienteNormalizado);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map((p) => `${labels[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
  .join("\n");

console.log(results);
