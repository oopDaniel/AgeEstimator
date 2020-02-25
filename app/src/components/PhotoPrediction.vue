<template>
  <div class="photo-prediction">
    <el-card
      class="prediction-card"
      v-for="[model, age] in ageList"
      :key="model"
    >
      {{ model }} model says {{ age }} years old
    </el-card>
  </div>
</template>

<script>
import * as R from 'ramda'

export default {
  name: 'PhotoPrediction',
  props: {
    ages: {
      type: Object,
      validator: R.compose(
        R.all(R.is(Number)),
        R.props(['cnn', 'regression', 'clustering'])
      )
    }
  },
  computed: {
    ageList() {
      return R.toPairs(this.ages)
    }
  }
}
</script>

<style scoped lang="scss">
.photo-prediction {
  display: flex;
  justify-content: center;
}

.prediction-card {
  min-height: 300px;
  max-width: 320px;
  flex: 1;
  margin-left: 3rem;
  margin-right: 3rem;
}
</style>
