<template>
  <div class="photo-uploader">
    <h1>{{ msg }}</h1>
    <section>
      <div>Upload your image below</div>
      <div class="desc">(square image size preferred)</div>
      <section class="card-section center">
        <el-card class="card-box">
          <el-upload
            :class="{
              'avatar-uploader': true,
              'post-upload': hasUploaded
            }"
            action="/api/predict_age"
            :show-file-list="false"
            :on-success="handleAvatarSuccess"
            :before-upload="beforeAvatarUpload"
            :multiple="false"
          >
            <i class="el-icon-plus avatar-uploader-icon"></i>
          </el-upload>
          <div v-if="imageUrl !== ''" class="preview-box">
            <img width="100%" :src="imageUrl" alt="" />
          </div>
        </el-card>
      </section>
      <PhotoPrediction
        class="prediction"
        v-if="hasUploaded"
        :ages="prediction.ages"
      />
    </section>
  </div>
</template>

<script>
import PhotoPrediction from './PhotoPrediction'

const VALID_FILE_TYPES = ['image/png', 'image/jpeg']
const UNSUPPORTED_MSG = 'Unsupported picture format. Please upload a JPG or PNG'
const TOO_LARGE_MSG = 'Picture size can not exceed 2MB'
const MAX_SIZE = 1024 ** 2

export default {
  name: 'PhotoUploader',
  props: {
    msg: {
      type: String,
      default: ''
    }
  },
  data: () => ({
    imageUrl: '',
    activeFileName: '',
    prediction: null
  }),
  computed: {
    hasUploaded() {
      return this.prediction !== null && this.prediction.ages !== undefined
    }
  },
  methods: {
    handleAvatarSuccess(prediction, file) {
      if (prediction.name === this.activeFileName) {
        this.prediction = prediction
        this.imageUrl = URL.createObjectURL(file.raw)
      }
    },
    beforeAvatarUpload({ type, size, name }) {
      this.prediction = null

      const isValidType = VALID_FILE_TYPES.includes(type)
      const isLt2M = size / MAX_SIZE < 2

      if (!isValidType) {
        this.$message.error(UNSUPPORTED_MSG)
      } else if (!isLt2M) {
        this.$message.error(TOO_LARGE_MSG)
      } else {
        this.activeFileName = name
      }

      return isValidType && isLt2M
    }
  },
  components: {
    PhotoPrediction
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
.photo-uploader {
  color: var(--title);
  padding-bottom: 6rem;
}

.desc {
  margin-top: 0.2rem;
  color: var(--desc);
  font-style: italic;
}

.card-section {
  margin-top: 2rem;
  flex-direction: column;
}
.card-box {
  position: relative;
  max-width: 400px;
  width: 320px;
  height: 320px;
  box-shadow: 0 2px 6px 0 rgba(0, 0, 0, 0.1);
  &:hover {
    box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.1);
  }
}

.preview-box {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 80%;
  max-width: 320px;
  cursor: pointer;
}

.prediction {
  margin-top: 2rem;
}
</style>

<style lang="scss">
.photo-uploader {
  .el-card__body {
    position: relative;
    height: 100%;
    padding: 0;
  }
  .avatar-uploader {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1;
  }

  .el-upload {
    border: 1px dashed #d9d9d9;
    border-radius: 6px;
    position: relative;
    cursor: pointer;
    transition: border-color 0.2s;
    &:hover {
      border-color: var(--theme1);
    }
  }

  .post-upload {
    .el-upload {
      border-color: transparent;
      &:hover {
        border-color: var(--theme1);
      }
    }
    .avatar-uploader-icon {
      color: transparent;
      &:hover {
        color: var(--desc);
      }
    }
  }

  .avatar-uploader-icon {
    font-size: 28px;
    color: var(--desc);
    width: 240px;
    height: 240px;
    line-height: 240px;
    text-align: center;
    transition: color 0.2s;
  }
  .avatar {
    width: 240px;
    height: 240px;
    display: block;
  }
}
</style>
