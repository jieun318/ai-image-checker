// generate-initial-model.js
// 초기 기본 모델(model.json & weights.bin)을 생성하는 유틸리티 스크립트

const fs = require('fs');
const path = require('path');

const modelDir = path.join(__dirname, 'model');
if (!fs.existsSync(modelDir)) {
  fs.mkdirSync(modelDir, { recursive: true });
}

// 가중치 명세
const weightSpecs = [
  { name: 'dense_1/kernel', shape: [1280, 128] },
  { name: 'dense_1/bias', shape: [128] },
  { name: 'dense_2/kernel', shape: [128, 64] },
  { name: 'dense_2/bias', shape: [64] },
  { name: 'dense_3/kernel', shape: [64, 2] },
  { name: 'dense_3/bias', shape: [2] }
];

let totalFloats = 0;
weightSpecs.forEach(w => {
  const count = w.shape.reduce((a, b) => a * b, 1);
  totalFloats += count;
});

// Float32Array 버퍼 할당 및 Xavier/He 초기화 난수 채우기
const floatBuffer = new Float32Array(totalFloats);
let offset = 0;

weightSpecs.forEach(w => {
  const count = w.shape.reduce((a, b) => a * b, 1);
  const fanIn = w.shape[0];
  const std = Math.sqrt(2.0 / fanIn);

  if (w.name.endsWith('/bias')) {
    // 바이어스는 0으로 초기화
    for (let i = 0; i < count; i++) {
      floatBuffer[offset + i] = 0.0;
    }
  } else {
    // 커널은 정규분포 가중치로 초기화
    for (let i = 0; i < count; i++) {
      // Box-Muller transform
      const u1 = Math.random() || 0.0001;
      const u2 = Math.random() || 0.0001;
      const randNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      floatBuffer[offset + i] = randNormal * std;
    }
  }
  offset += count;
});

// 바이너리 파일 저장
const weightsBinPath = path.join(modelDir, 'weights.bin');
fs.writeFileSync(weightsBinPath, Buffer.from(floatBuffer.buffer));

// model.json 구성
const modelJson = {
  format: 'layers-model',
  generatedBy: 'TensorFlow.js tfjs-core v4.22.0',
  convertedBy: null,
  modelTopology: {
    class_name: 'Sequential',
    config: {
      name: 'transfer_model',
      layers: [
        {
          class_name: 'Dense',
          config: {
            units: 128,
            activation: 'relu',
            use_bias: true,
            kernel_initializer: {
              class_name: 'VarianceScaling',
              config: { scale: 1, mode: 'fan_in', distribution: 'normal', seed: null }
            },
            bias_initializer: { class_name: 'Zeros', config: {} },
            name: 'dense_1',
            trainable: true,
            batch_input_shape: [null, 1280],
            dtype: 'float32'
          }
        },
        {
          class_name: 'Dropout',
          config: {
            rate: 0.25,
            name: 'dropout_1',
            trainable: true
          }
        },
        {
          class_name: 'Dense',
          config: {
            units: 64,
            activation: 'relu',
            use_bias: true,
            kernel_initializer: {
              class_name: 'VarianceScaling',
              config: { scale: 1, mode: 'fan_in', distribution: 'normal', seed: null }
            },
            bias_initializer: { class_name: 'Zeros', config: {} },
            name: 'dense_2',
            trainable: true,
            dtype: 'float32'
          }
        },
        {
          class_name: 'Dense',
          config: {
            units: 2,
            activation: 'softmax',
            use_bias: true,
            kernel_initializer: {
              class_name: 'VarianceScaling',
              config: { scale: 1, mode: 'fan_in', distribution: 'normal', seed: null }
            },
            bias_initializer: { class_name: 'Zeros', config: {} },
            name: 'dense_3',
            trainable: true,
            dtype: 'float32'
          }
        }
      ]
    },
    keras_version: 'tfjs-layers 4.22.0',
    backend: 'tensor_flow.js'
  },
  weightsManifest: [
    {
      paths: ['weights.bin'],
      weights: weightSpecs.map(w => ({
        name: w.name,
        shape: w.shape,
        dtype: 'float32'
      }))
    }
  ]
};

const modelJsonPath = path.join(modelDir, 'model.json');
fs.writeFileSync(modelJsonPath, JSON.stringify(modelJson, null, 2), 'utf-8');

console.log(`성공: 기본 모델 파일 생성 완료!`);
console.log(`- ${modelJsonPath}`);
console.log(`- ${weightsBinPath} (${(floatBuffer.byteLength / 1024).toFixed(1)} KB)`);
